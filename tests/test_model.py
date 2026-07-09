"""Self-check for model.py — no dataset/training required."""
import torch

from model import GradCAM, build_model


def test_build_model_output_shape():
    model = build_model(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 2)


def test_gradcam_output_shape():
    model = build_model(pretrained=False)
    cam = GradCAM(model)
    x = torch.randn(1, 3, 224, 224)
    heatmap = cam(x, class_idx=0)
    assert heatmap.min() >= 0 and heatmap.max() <= 1
    assert heatmap.ndim == 2


if __name__ == "__main__":
    test_build_model_output_shape()
    test_gradcam_output_shape()
    print("ok")
