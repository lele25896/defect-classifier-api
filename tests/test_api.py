"""API contract test using an untrained dummy checkpoint — verifies the
request/response wiring, not model accuracy (that needs real MVTec AD data,
see data/prepare_mvtec.py + train.py).
"""
import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from model import OODStats, build_model, build_transform, extract_features


def _make_dummy_checkpoint(tmp_path, with_ood_stats=False):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model = build_model(pretrained=False)
    model.eval()  # matches app.main.load_models() — BatchNorm must use running stats, not batch stats
    torch.save(model.state_dict(), models_dir / "bottle_resnet18.pt")

    if with_ood_stats:
        transform = build_transform(train=False)
        bank_images = [Image.new("RGB", (224, 224), color=(g, g, g)) for g in range(100, 160, 4)]
        feats = torch.cat([extract_features(model, transform(im).unsqueeze(0)) for im in bank_images])
        OODStats.fit(feats).save(models_dir / "bottle_ood_stats.pt")

    return models_dir


def _dummy_image_bytes(color=(128, 128, 128)):
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=color).save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MODELS_DIR", _make_dummy_checkpoint(tmp_path))
    with TestClient(main_module.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["categories"] == ["bottle"]


def test_predict_known_category(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MODELS_DIR", _make_dummy_checkpoint(tmp_path))
    with TestClient(main_module.app) as client:
        r = client.post(
            "/predict",
            params={"category": "bottle"},
            files={"file": ("test.png", _dummy_image_bytes(), "image/png")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "defective" in body and "confidence" in body
        assert body["ood"] is None  # no *_ood_stats.pt sidecar for this checkpoint


def test_predict_ood_flag_present_when_stats_available(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MODELS_DIR", _make_dummy_checkpoint(tmp_path, with_ood_stats=True))
    with TestClient(main_module.app) as client:
        r = client.post(
            "/predict",
            params={"category": "bottle"},
            files={"file": ("test.png", _dummy_image_bytes(), "image/png")},
        )
        assert r.status_code == 200
        assert r.json()["ood"] is False  # matches the bank's gray-image distribution


def test_predict_ood_true_for_out_of_distribution_image(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MODELS_DIR", _make_dummy_checkpoint(tmp_path, with_ood_stats=True))
    with TestClient(main_module.app) as client:
        r = client.post(
            "/predict",
            params={"category": "bottle"},
            files={"file": ("test.png", _dummy_image_bytes(color=(255, 0, 0)), "image/png")},
        )
        assert r.status_code == 200
        assert r.json()["ood"] is True  # solid red is nothing like the gray-scale bank


def test_predict_heatmap(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MODELS_DIR", _make_dummy_checkpoint(tmp_path))
    with TestClient(main_module.app) as client:
        r = client.post(
            "/predict/heatmap",
            params={"category": "bottle"},
            files={"file": ("test.png", _dummy_image_bytes(), "image/png")},
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/png"
        assert len(r.content) > 0


def test_predict_unknown_category_404(tmp_path, monkeypatch):
    monkeypatch.setattr(main_module, "MODELS_DIR", _make_dummy_checkpoint(tmp_path))
    with TestClient(main_module.app) as client:
        r = client.post(
            "/predict",
            params={"category": "nonexistent"},
            files={"file": ("test.png", _dummy_image_bytes(), "image/png")},
        )
        assert r.status_code == 404
