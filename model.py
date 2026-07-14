"""ResNet18 transfer-learning classifier for MVTec AD (good vs defective) + Grad-CAM."""
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


def build_transform(train: bool) -> transforms.Compose:
    ops = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    if train:
        # ponytail: light augmentation — MVTec AD train splits are small (~200-400 imgs/category)
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(ops)


def build_model(pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    # ponytail: freeze everything except layer4 + fc — small dataset, fine-tuning
    # the whole net would overfit and is slower for no accuracy gain here.
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("layer4") or name.startswith("fc")
    model.fc = nn.Linear(model.fc.in_features, 2)  # 0=good, 1=defective
    return model


class GradCAM:
    """Minimal Grad-CAM on resnet18's last conv block (layer4).

    ponytail: hand-rolled instead of adding the `grad-cam` pip dependency —
    it's ~20 lines for a single fixed architecture, not worth a new package.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None
        model.layer4.register_forward_hook(self._save_activations)
        model.layer4.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int) -> torch.Tensor:
        """Returns a [H, W] heatmap in [0, 1] for input x [1, 3, H, W]."""
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        cam = (weights * self.activations).sum(dim=1).squeeze(0)  # [h, w]
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Pooled layer4 features [B, 512] (avgpool output, fc's input) — used for OOD scoring."""
    feats = {}
    h = model.avgpool.register_forward_hook(lambda m, i, o: feats.__setitem__("v", o))
    with torch.no_grad():
        model(x)
    h.remove()
    return feats["v"].flatten(1)


OOD_KNN_K = 3
OOD_PERCENTILE = 99  # threshold = this percentile of in-distribution self-distances


class OODStats:
    """Per-category feature bank (pooled features of every local image) used for
    k-NN distance OOD scoring: a new image is OOD if it sits farther from its k
    nearest in-distribution neighbors than in-distribution images ever sit from
    each other.

    Flags inputs whose visual domain doesn't match training (e.g. a render vs
    MVTec's real photographed bottles) — softmax confidence can't catch this, a
    confident wrong prediction on an out-of-domain image looks identical to a
    confident right one (see 2026-07-14 false positive: a bottle render hit 0.99
    confidence "defective" with nothing actually wrong with it).

    ponytail: tried mean/std z-score first (rung 5, cheaper) — didn't separate
    the known false positive from real defects (a real "contamination" image
    scored a higher z than the fake render). k-NN against the raw feature bank
    is rung 6 but was needed for it to actually work; validated against 3 real
    MVTec defect images + the known false positive before picking the threshold.
    """

    def __init__(self, bank: torch.Tensor, threshold: float):
        self.bank = bank
        self.threshold = threshold

    def distance(self, feat: torch.Tensor) -> float:
        d = torch.cdist(feat.reshape(1, -1), self.bank).squeeze(0)
        return float(d.topk(OOD_KNN_K, largest=False).values.mean())

    def is_ood(self, feat: torch.Tensor) -> bool:
        return self.distance(feat) > self.threshold

    def save(self, path) -> None:
        torch.save({"bank": self.bank, "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path) -> "OODStats":
        d = torch.load(path, weights_only=True, map_location="cpu")
        return cls(d["bank"], d["threshold"])

    @classmethod
    def fit(cls, feats: torch.Tensor) -> "OODStats":
        d = torch.cdist(feats, feats)
        d.fill_diagonal_(float("inf"))
        loo_dist = d.topk(OOD_KNN_K, largest=False).values.mean(dim=1)
        threshold = float(loo_dist.quantile(OOD_PERCENTILE / 100))
        return cls(feats, threshold)


def colorize_heatmap(cam: np.ndarray) -> np.ndarray:
    """[H, W] in [0, 1] -> [H, W, 3] uint8, blue (cold) to red (hot).

    ponytail: 4-stop numpy interpolation instead of matplotlib — a single
    fixed colormap doesn't need a 50MB plotting library in the API image.
    """
    stops = [0, 1 / 3, 2 / 3, 1]
    r = np.interp(cam, stops, [0, 0, 1, 1])
    g = np.interp(cam, stops, [0, 1, 1, 0])
    b = np.interp(cam, stops, [1, 1, 0, 0])
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
