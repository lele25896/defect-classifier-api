"""Train a good/defective classifier for one MVTec AD category.

MVTec AD's own train/ split has only "good" images (it's built for
unsupervised anomaly detection). We want a supervised classifier, so we
pool train/good + every image under test/ (good and all defect types) and
do our own stratified train/val/test split — same discipline as the Fraud
Detection project: split first, only fit/augment after.

Usage: python train.py --category bottle --data-dir data --epochs 15
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from model import build_model, build_transform


def collect_samples(category_dir: Path) -> list[tuple[Path, int]]:
    samples = []
    for img_path in (category_dir / "train" / "good").glob("*.png"):
        samples.append((img_path, 0))
    for defect_dir in (category_dir / "test").iterdir():
        label = 0 if defect_dir.name == "good" else 1
        for img_path in defect_dir.glob("*.png"):
            samples.append((img_path, label))
    return samples


class MVTecDataset(Dataset):
    def __init__(self, samples: list[tuple[Path, int]], train: bool):
        self.samples = samples
        self.transform = build_transform(train)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * len(y)
        preds += logits.argmax(1).tolist()
        labels += y.tolist()
    return total_loss / len(labels), preds, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category_dir = Path(args.data_dir) / args.category
    samples = collect_samples(category_dir)
    labels = [label for _, label in samples]

    train_s, temp_s = train_test_split(samples, test_size=0.4, stratify=labels, random_state=42)
    val_s, test_s = train_test_split(
        temp_s, test_size=0.5, stratify=[l for _, l in temp_s], random_state=42
    )

    train_loader = DataLoader(MVTecDataset(train_s, train=True), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(MVTecDataset(val_s, train=False), batch_size=args.batch_size)
    test_loader = DataLoader(MVTecDataset(test_s, train=False), batch_size=args.batch_size)

    model = build_model().to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best_val_loss = float("inf")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    ckpt_path = models_dir / f"{args.category}_resnet18.pt"

    for epoch in range(args.epochs):
        train_loss, _, _ = run_epoch(model, train_loader, device, optimizer)
        val_loss, val_preds, val_labels = run_epoch(model, val_loader, device)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    _, test_preds, test_labels = run_epoch(model, test_loader, device)
    report = classification_report(test_labels, test_preds, target_names=["good", "defective"], output_dict=True)
    print(classification_report(test_labels, test_preds, target_names=["good", "defective"]))

    with open(models_dir / f"{args.category}_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
