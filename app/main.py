import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from PIL import Image

from model import GradCAM, OODStats, build_model, build_transform, colorize_heatmap, extract_features

logging.basicConfig(level=logging.INFO, format='{"level":"%(levelname)s","msg":"%(message)s"}')
logger = logging.getLogger("defect-api")

MODELS_DIR = Path(__file__).parent.parent / "models"
TRANSFORM = build_transform(train=False)


def load_models() -> dict[str, torch.nn.Module]:
    models = {}
    for ckpt in MODELS_DIR.glob("*_resnet18.pt"):
        category = ckpt.stem.removesuffix("_resnet18")
        model = build_model(pretrained=False)
        model.load_state_dict(torch.load(ckpt, weights_only=True, map_location="cpu"))
        model.eval()
        models[category] = model
        logger.info(f"loaded model for category={category}")
    return models


def load_ood_stats() -> dict[str, OODStats]:
    """Sidecar per category — optional, e.g. absent for a dummy test checkpoint."""
    stats = {}
    for path in MODELS_DIR.glob("*_ood_stats.pt"):
        category = path.stem.removesuffix("_ood_stats")
        stats[category] = OODStats.load(path)
    return stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_models()
    app.state.gradcams = {cat: GradCAM(m) for cat, m in app.state.models.items()}
    app.state.ood_stats = load_ood_stats()
    yield


app = FastAPI(lifespan=lifespan)


def _load_image(file: UploadFile) -> tuple[Image.Image, torch.Tensor]:
    img = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    return img, TRANSFORM(img).unsqueeze(0)


def _get_model(category: str) -> torch.nn.Module:
    model = app.state.models.get(category)
    if model is None:
        available = list(app.state.models.keys())
        raise HTTPException(404, f"unknown category '{category}', available: {available}")
    return model


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "categories": list(app.state.models.keys())}


@app.post("/predict")
def predict(category: str, file: UploadFile):
    model = _get_model(category)
    _, x = _load_image(file)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    defective = bool(probs[1] > probs[0])

    ood_stats = app.state.ood_stats.get(category)
    ood = ood_stats.is_ood(extract_features(model, x)) if ood_stats else None

    logger.info(
        f"category={category} defective={defective} confidence={float(probs.max()):.3f} ood={ood}"
    )
    return {
        "category": category,
        "defective": defective,
        "confidence": float(probs.max()),
        "ood": ood,
    }


@app.post("/predict/heatmap")
def predict_heatmap(category: str, file: UploadFile):
    model = _get_model(category)
    img, x = _load_image(file)
    with torch.no_grad():
        class_idx = int(torch.softmax(model(x), dim=1).argmax())
    cam = app.state.gradcams[category](x, class_idx)  # needs grad, no no_grad here
    heatmap_rgb = colorize_heatmap(cam.numpy())
    heatmap_img = Image.fromarray(heatmap_rgb).resize(img.size)
    overlay = Image.blend(img, heatmap_img, alpha=0.45)

    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
