"""Convert MOT17 annotations to COCO detection JSON for RT-DETR fine-tuning.

Expected MOT17-like sequence structure:

    MOT17-02-FRCNN/
      img1/000001.jpg
      gt/gt.txt
      seqinfo.ini

The script keeps one category, person, with category_id=0 because RT-DETR is
configured with num_classes=1 for MOT17.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SequenceInfo:
    name: str
    path: Path
    image_dir: Path
    image_ext: str
    width: int
    height: int
    length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mot-root",
        type=Path,
        default=Path("dataset/mot17/raw"),
        help="Root containing MOT17 sequences or train/test folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("dataset/mot17/annotations"),
        help="Directory where train.json and val.json will be written.",
    )
    parser.add_argument(
        "--detector-variant",
        choices=["FRCNN", "DPM", "SDP", "all"],
        default="FRCNN",
        help="MOT17 repeats sequences for detector variants. Use one variant to avoid duplicate frames.",
    )
    parser.add_argument(
        "--split",
        choices=["half", "sequence"],
        default="half",
        help="half: split every sequence by frame index. sequence: hold out --val-seqs.",
    )
    parser.add_argument(
        "--val-seqs",
        nargs="*",
        default=["MOT17-02-FRCNN", "MOT17-09-FRCNN"],
        help="Validation sequences for --split sequence.",
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
        help="Drop boxes with visibility below this value when visibility column exists.",
    )
    parser.add_argument(
        "--keep-unlabeled-images",
        action="store_true",
        help="Keep frames without valid boxes. By default they are skipped.",
    )
    return parser.parse_args()


def read_seqinfo(seq_dir: Path) -> tuple[str, str, int, int, int]:
    seqinfo = seq_dir / "seqinfo.ini"
    if seqinfo.exists():
        parser = configparser.ConfigParser()
        parser.read(seqinfo)
        section = parser["Sequence"]
        image_dir = section.get("imDir", "img1")
        image_ext = section.get("imExt", ".jpg")
        width = section.getint("imWidth")
        height = section.getint("imHeight")
        length = section.getint("seqLength")
        return image_dir, image_ext, width, height, length

    image_dir = "img1"
    images = sorted((seq_dir / image_dir).glob("*"))
    if not images:
        raise FileNotFoundError(f"No images found in {seq_dir / image_dir}")

    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required when seqinfo.ini is missing.") from exc

    with Image.open(images[0]) as im:
        width, height = im.size
    return image_dir, images[0].suffix, width, height, len(images)


def find_sequences(mot_root: Path, detector_variant: str) -> list[SequenceInfo]:
    candidates = [p for p in mot_root.rglob("gt.txt") if p.parent.name == "gt"]
    sequences: list[SequenceInfo] = []

    for gt_file in candidates:
        seq_dir = gt_file.parent.parent
        if detector_variant != "all" and not seq_dir.name.endswith(f"-{detector_variant}"):
            continue

        image_dir_name, image_ext, width, height, length = read_seqinfo(seq_dir)
        image_dir = seq_dir / image_dir_name
        if not image_dir.exists():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")

        sequences.append(
            SequenceInfo(
                name=seq_dir.name,
                path=seq_dir,
                image_dir=image_dir,
                image_ext=image_ext,
                width=width,
                height=height,
                length=length,
            )
        )

    sequences.sort(key=lambda s: s.name)
    if not sequences:
        raise FileNotFoundError(
            f"No MOT17 sequences found under {mot_root} for detector_variant={detector_variant}"
        )
    return sequences


def frame_file(seq: SequenceInfo, frame_id: int) -> Path:
    return seq.image_dir / f"{frame_id:06d}{seq.image_ext}"


def relative_image_path(mot_root: Path, image_path: Path) -> str:
    return image_path.relative_to(mot_root).as_posix()


def load_mot_annotations(
    seq: SequenceInfo,
    include_classes: set[int],
    min_visibility: float,
) -> dict[int, list[dict]]:
    annotations_by_frame: dict[int, list[dict]] = {}
    gt_file = seq.path / "gt" / "gt.txt"

    with gt_file.open(newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 7:
                continue

            frame_id = int(float(row[0]))
            x = float(row[2])
            y = float(row[3])
            w = float(row[4])
            h = float(row[5])
            conf = int(float(row[6]))
            class_id = int(float(row[7])) if len(row) > 7 else 1
            visibility = float(row[8]) if len(row) > 8 else 1.0

            if conf != 1:
                continue
            if class_id not in include_classes:
                continue
            if visibility < min_visibility:
                continue
            if w <= 1 or h <= 1:
                continue

            x = max(0.0, min(x, seq.width - 1.0))
            y = max(0.0, min(y, seq.height - 1.0))
            w = max(0.0, min(w, seq.width - x))
            h = max(0.0, min(h, seq.height - y))
            if w <= 1 or h <= 1:
                continue

            annotations_by_frame.setdefault(frame_id, []).append(
                {
                    "bbox": [round(x, 3), round(y, 3), round(w, 3), round(h, 3)],
                    "area": round(w * h, 3),
                    "iscrowd": 0,
                    "category_id": 0,
                    "mot_track_id": int(float(row[1])),
                    "mot_class_id": class_id,
                    "visibility": visibility,
                }
            )

    return annotations_by_frame


def selected_frames(seq: SequenceInfo, split: str, subset: str, val_seq_names: set[str]) -> Iterable[int]:
    if split == "sequence":
        is_val = seq.name in val_seq_names
        if (subset == "val") != is_val:
            return []
        return range(1, seq.length + 1)

    midpoint = seq.length // 2
    if subset == "train":
        return range(1, midpoint + 1)
    return range(midpoint + 1, seq.length + 1)


def build_coco(
    mot_root: Path,
    sequences: list[SequenceInfo],
    split: str,
    subset: str,
    val_seq_names: set[str],
    include_classes: set[int],
    min_visibility: float,
    keep_unlabeled_images: bool,
) -> dict:
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    for seq in sequences:
        by_frame = load_mot_annotations(seq, include_classes, min_visibility)
        for frame_id in selected_frames(seq, split, subset, val_seq_names):
            frame_annotations = by_frame.get(frame_id, [])
            if not frame_annotations and not keep_unlabeled_images:
                continue

            image_path = frame_file(seq, frame_id)
            if not image_path.exists():
                continue

            images.append(
                {
                    "id": image_id,
                    "file_name": relative_image_path(mot_root, image_path),
                    "height": seq.height,
                    "width": seq.width,
                    "frame_id": frame_id,
                    "video_id": seq.name,
                }
            )

            for ann in frame_annotations:
                ann = dict(ann)
                ann["id"] = annotation_id
                ann["image_id"] = image_id
                annotations.append(ann)
                annotation_id += 1

            image_id += 1

    return {
        "info": {
            "description": "MOT17 converted to COCO detection format for RT-DETR fine-tuning",
            "person_category_id": 0,
        },
        "licenses": [],
        "categories": [{"id": 0, "name": "person", "supercategory": "person"}],
        "images": images,
        "annotations": annotations,
    }


def main() -> None:
    args = parse_args()
    mot_root = args.mot_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sequences = find_sequences(mot_root, args.detector_variant)
    include_classes = set(args.include_classes)
    val_seq_names = set(args.val_seqs)

    for subset in ("train", "val"):
        coco = build_coco(
            mot_root=mot_root,
            sequences=sequences,
            split=args.split,
            subset=subset,
            val_seq_names=val_seq_names,
            include_classes=include_classes,
            min_visibility=args.min_visibility,
            keep_unlabeled_images=args.keep_unlabeled_images,
        )
        out_file = out_dir / f"{subset}.json"
        with out_file.open("w") as f:
            json.dump(coco, f)

        print(
            f"{subset}: {len(coco['images'])} images, "
            f"{len(coco['annotations'])} boxes -> {out_file}"
        )

    print("Sequences used:")
    for seq in sequences:
        print(f"  - {seq.name}: {seq.length} frames, {seq.width}x{seq.height}")


if __name__ == "__main__":
    main()
