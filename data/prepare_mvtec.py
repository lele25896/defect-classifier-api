"""Extract MVTec AD category archives into data/<category>/.

Category archives are public direct downloads (no login/license
click-through), listed per-category on
https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
(mirrored on mydrive.ch). Manual step:

1. Download the categories you want (e.g. bottle.tar.xz, hazelnut.tar.xz)
   from the page above and drop them in data/downloads/
2. Run: python data/prepare_mvtec.py bottle hazelnut
"""
import sys
import tarfile
from pathlib import Path

DOWNLOADS = Path(__file__).parent / "downloads"
DATA_DIR = Path(__file__).parent


def extract(category: str) -> None:
    archive = DOWNLOADS / f"{category}.tar.xz"
    if not archive.exists():
        raise SystemExit(f"missing {archive} — download it first, see module docstring")
    target = DATA_DIR / category
    if target.exists():
        print(f"{category}: already extracted, skipping")
        return
    with tarfile.open(archive) as tar:
        tar.extractall(DATA_DIR)
    print(f"{category}: extracted to {target}")


if __name__ == "__main__":
    categories = sys.argv[1:] or ["bottle", "hazelnut"]
    for c in categories:
        extract(c)
