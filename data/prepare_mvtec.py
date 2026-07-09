"""Extract MVTec AD category archives into data/<category>/.

MVTec AD requires accepting a license on their site before download — no
stable direct-download URL to script around that (and scripting past a
license gate isn't something to automate). Manual step:

1. https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
2. Download the categories you want (e.g. bottle.tar.xz, hazelnut.tar.xz)
3. Drop them in data/downloads/
4. Run: python data/prepare_mvtec.py bottle hazelnut
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
