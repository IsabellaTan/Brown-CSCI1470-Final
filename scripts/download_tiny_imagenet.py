# python scripts/download_tiny_imagenet.py

#!/usr/bin/env python3
import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def download_file(url: str, output_path: Path) -> None:
    print(f"Downloading Tiny-ImageNet to: {output_path}")
    with urllib.request.urlopen(url) as response, output_path.open("wb") as f:
        shutil.copyfileobj(response, f)


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract Tiny-ImageNet")
    parser.add_argument(
        "target_dir",
        nargs="?",
        default="data",
        help="Directory to store tiny-imagenet-200 (default: data)",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    zip_path = target_dir / "tiny-imagenet-200.zip"
    data_dir = target_dir / "tiny-imagenet-200"

    if (data_dir / "train").exists() and (data_dir / "val").exists():
        print(f"Dataset already exists: {data_dir}")
        print(f"Train with: bash scripts/train_baseline.sh {data_dir}")
        return 0

    if not zip_path.exists():
        try:
            download_file(URL, zip_path)
        except Exception as exc:
            print(f"Download failed: {exc}", file=sys.stderr)
            return 1

    try:
        extract_zip(zip_path, target_dir)
    except zipfile.BadZipFile:
        print("Error: downloaded zip is corrupted.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Extraction failed: {exc}", file=sys.stderr)
        return 1

    print(f"Done: {data_dir}")
    print(f"Train with: bash scripts/train_baseline.sh {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
