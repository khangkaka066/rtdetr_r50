"""Prepare a MOT17 dataset path for RT-DETR fine-tuning.

This creates a stable virtual dataset path used by configs:

    dataset/mot17/raw -> /your/real/MOT17/train/path

Then it converts MOT17 gt.txt annotations to COCO JSON:

    dataset/mot17/annotations/train.json
    dataset/mot17/annotations/val.json

Usage:

    python tools/prepare_mot17_dataset.py /kaggle/input/.../MOT17/train
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_root",
        type=Path,
        help="Real MOT17 train directory. Example: /kaggle/input/.../MOT17/train",
    )
    parser.add_argument(
        "--virtual-root",
        type=Path,
        default=Path("dataset/mot17/raw"),
        help="Stable virtual path used by configs.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("dataset/mot17/annotations"),
        help="Where COCO train.json and val.json are written.",
    )
    parser.add_argument(
        "--detector-variant",
        choices=["FRCNN", "DPM", "SDP", "all"],
        default="FRCNN",
        help="MOT17 detector variant to use. FRCNN avoids duplicated sequence variants.",
    )
    parser.add_argument(
        "--split",
        choices=["half", "sequence"],
        default="half",
        help="How to create train/val JSON from MOT17 train sequences.",
    )
    parser.add_argument(
        "--include-classes",
        type=int,
        nargs="+",
        default=[1],
        help="MOT class ids to keep. 1 is pedestrian in MOT17.",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.1,
        help="Drop boxes with visibility below this value.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing virtual-root path if it points elsewhere.",
    )
    return parser.parse_args()


def ensure_link(source: Path, link: Path, force: bool) -> None:
    source = source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Dataset path must be a directory: {source}")

    link.parent.mkdir(parents=True, exist_ok=True)

    if link.is_symlink():
        current = link.resolve()
        if current == source:
            print(f"Virtual dataset path already OK: {link} -> {source}")
            return
        if not force:
            raise FileExistsError(
                f"{link} already points to {current}. Re-run with --force to replace it."
            )
        link.unlink()
    elif link.exists():
        if not force:
            raise FileExistsError(
                f"{link} already exists and is not a symlink. Re-run with --force to replace it."
            )
        if link.is_dir():
            shutil.rmtree(link)
        else:
            link.unlink()

    os.symlink(source, link)
    print(f"Created virtual dataset path: {link} -> {source}")


def run_converter(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(Path(__file__).with_name("convert_mot17_to_coco.py")),
        "--mot-root",
        str(args.virtual_root),
        "--out-dir",
        str(args.annotations_dir),
        "--detector-variant",
        args.detector_variant,
        "--split",
        args.split,
        "--include-classes",
        *[str(x) for x in args.include_classes],
        "--min-visibility",
        str(args.min_visibility),
    ]
    print("Converting MOT17 to COCO:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    ensure_link(args.dataset_root, args.virtual_root, args.force)
    run_converter(args)

    print("\nReady to train:")
    print(
        "python tools/train.py "
        "-c configs/rtdetr/rtdetr_r50vd_6x_mot17.yml "
        "-t https://github.com/lyuwenyu/storage/releases/download/v0.1/"
        "rtdetr_r50vd_6x_coco_from_paddle.pth "
        "--amp"
    )


if __name__ == "__main__":
    main()
