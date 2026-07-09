"""API contract test using an untrained dummy checkpoint — verifies the
request/response wiring, not model accuracy (that needs real MVTec AD data,
see data/prepare_mvtec.py + train.py).
"""
import io

import torch
from fastapi.testclient import TestClient
from PIL import Image

import app.main as main_module
from model import build_model


def _make_dummy_checkpoint(tmp_path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    torch.save(build_model(pretrained=False).state_dict(), models_dir / "bottle_resnet18.pt")
    return models_dir


def _dummy_image_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (224, 224), color=(128, 128, 128)).save(buf, format="PNG")
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
