"""Self-check for model.py — no dataset/training required."""
import torch

from model import GradCAM, OODStats, build_model, extract_features


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


def test_ood_stats_flags_far_input_not_near_input():
    torch.manual_seed(0)
    bank = torch.randn(20, 512)  # synthetic "in-distribution" feature bank
    stats = OODStats.fit(bank)

    near = bank[0] + torch.randn(512) * 0.01
    far = torch.full((512,), 100.0)  # nothing like the bank -> should be OOD
    assert not stats.is_ood(near)
    assert stats.is_ood(far)


def test_extract_features_shape():
    model = build_model(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    feats = extract_features(model, x)
    assert feats.shape == (2, 512)


if __name__ == "__main__":
    test_build_model_output_shape()
    test_gradcam_output_shape()
    test_ood_stats_flags_far_input_not_near_input()
    test_extract_features_shape()
    print("ok")
