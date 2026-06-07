import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from PIL import Image, ImageDraw, ImageFont
import torch

from src.tracker import HybridMOTTracker, RTDETRDetector
from src.tracker.box_ops import cxcywh_to_xyxy


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_image_dir(source):
    source = Path(source)
    if (source / "img1").is_dir():
        return source / "img1"
    return source


def list_frames(source):
    image_dir = find_image_dir(source)
    frames = [p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    return sorted(frames)


def draw_tracks(image, tracks):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for track in tracks:
        x1, y1, x2, y2 = cxcywh_to_xyxy(track.bbox)
        color = "lime" if not track.is_missing else "yellow"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, max(0, y1 - 12)), f"id {track.track_id}", fill=color, font=font)
    return image


def write_mot_row(handle, row):
    frame, track_id, x, y, w, h, score, a, b, c = row
    handle.write(
        f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{score:.4f},{a},{b},{c}\n"
    )


def main(args):
    frames = list_frames(args.source)
    if not frames:
        raise FileNotFoundError(f"No images found in {args.source}")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    first = Image.open(frames[0]).convert("RGB")
    detector = RTDETRDetector(
        config=args.config,
        checkpoint=args.resume,
        device=device,
        image_size=args.image_size,
        score_threshold=args.det_score,
        person_label=None if args.all_classes else args.person_label,
        amp=args.amp,
    )
    tracker = HybridMOTTracker(
        image_size=first.size,
        device=device,
        max_age=args.max_age,
        min_hits=args.min_hits,
        score_threshold=args.track_score,
        lambda_motion=args.lambda_motion,
        lambda_iou=args.lambda_iou,
        lambda_app=args.lambda_app,
        use_neural_motion=not args.disable_neural_motion,
        motion_checkpoint=args.motion_checkpoint,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tracking {len(frames)} frames from {find_image_dir(args.source)}")
    print(f"Writing MOT results to {out_path}")
    with out_path.open("w") as handle:
        for idx, frame_path in enumerate(frames, start=1):
            image = Image.open(frame_path).convert("RGB")
            detections = detector(image)
            tracks = tracker.update(detections)
            rows = []
            for track in tracks:
                if track.is_missing and not args.write_missing:
                    continue
                rows.append(track.mot_row(idx))
            for row in rows:
                write_mot_row(handle, row)

            if vis_dir:
                draw_tracks(image, tracks).save(vis_dir / frame_path.name)

            if idx == 1 or idx % args.log_step == 0 or idx == len(frames):
                print(
                    f"Frame {idx}/{len(frames)} "
                    f"detections={len(detections)} active_tracks={len(tracks)} written={len(rows)}",
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RT-DETR + hybrid Kalman/xLSTM/LNN MOT tracking.")
    parser.add_argument("--source", required=True, help="MOT sequence directory, img1 directory, or image directory")
    parser.add_argument("-c", "--config", default="configs/rtdetr/rtdetr_r50vd_6x_mot17.yml")
    parser.add_argument("-r", "--resume", required=True, help="RT-DETR detector checkpoint")
    parser.add_argument("--output", default="output/tracks.txt")
    parser.add_argument("--vis-dir", default="", help="optional directory for annotated frames")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--det-score", type=float, default=0.35)
    parser.add_argument("--track-score", type=float, default=0.35)
    parser.add_argument("--person-label", type=int, default=0)
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--lambda-motion", type=float, default=0.25)
    parser.add_argument("--lambda-iou", type=float, default=0.65)
    parser.add_argument("--lambda-app", type=float, default=0.10)
    parser.add_argument("--motion-checkpoint", default="", help="optional trained xLSTM/LNN residual checkpoint")
    parser.add_argument("--disable-neural-motion", action="store_true")
    parser.add_argument("--write-missing", action="store_true", help="also write predicted boxes for missing tracks")
    parser.add_argument("--log-step", type=int, default=50)
    main(parser.parse_args())
