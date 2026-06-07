import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

from src.tracker import ByteTrackXLSTMTracker, HybridMOTTracker, RTDETRDetector
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


def draw_tracks(image, tracks, show_missing=False):
    image = image.copy()
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for track in tracks:
        if track.is_missing and not show_missing:
            continue
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


def read_sequence_fps(source, default=30.0):
    source = Path(source)
    seqinfo = source / "seqinfo.ini"
    if not seqinfo.exists() and source.name == "img1":
        seqinfo = source.parent / "seqinfo.ini"
    if not seqinfo.exists():
        return float(default)

    for line in seqinfo.read_text().splitlines():
        if line.startswith("frameRate="):
            try:
                return float(line.split("=", 1)[1])
            except ValueError:
                return float(default)
    return float(default)


class VideoWriter:
    def __init__(self, path, fps):
        self.path = Path(path) if path else None
        self.fps = float(fps)
        self.writer = None

    def write(self, image):
        if self.path is None:
            return
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("Install opencv-python-headless to write video output.") from exc

        frame = np.asarray(image.convert("RGB"))
        height, width = frame.shape[:2]
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.path), fourcc, self.fps, (width, height))
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None


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
        color_embedding=not args.disable_color_embedding,
        nms_iou_threshold=args.nms_iou,
    )
    use_neural_motion = (args.enable_neural_motion or bool(args.motion_checkpoint)) and not args.disable_neural_motion
    if args.tracker == "byte_xlstm":
        tracker = ByteTrackXLSTMTracker(
            image_size=first.size,
            device=device,
            track_thresh=args.track_score,
            low_thresh=args.low_track_score,
            new_track_thresh=args.new_track_score,
            match_thresh=args.match_cost_threshold,
            second_match_thresh=args.low_match_cost_threshold,
            track_buffer=args.max_age,
            min_hits=args.min_hits,
            nms_duplicate_iou=args.duplicate_iou,
            lambda_iou=args.lambda_iou,
            lambda_app=args.lambda_app,
            fuse_score=not args.disable_fuse_score,
            use_xlstm=use_neural_motion,
            motion_backend=args.motion_backend,
            motion_checkpoint=args.motion_checkpoint,
        )
    else:
        tracker = HybridMOTTracker(
            image_size=first.size,
            device=device,
            max_age=args.max_age,
            min_hits=args.min_hits,
            score_threshold=args.track_score,
            low_score_threshold=args.low_track_score,
            lambda_motion=args.lambda_motion,
            lambda_iou=args.lambda_iou,
            lambda_app=args.lambda_app,
            duplicate_iou_threshold=args.duplicate_iou,
            match_cost_threshold=args.match_cost_threshold,
            low_match_cost_threshold=args.low_match_cost_threshold,
            fuse_score=not args.disable_fuse_score,
            use_neural_motion=use_neural_motion,
            motion_backend=args.motion_backend,
            motion_checkpoint=args.motion_checkpoint,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)
    fps = args.video_fps if args.video_fps > 0 else read_sequence_fps(args.source)
    video_writer = VideoWriter(args.video_output, fps)

    print(f"Tracking {len(frames)} frames from {find_image_dir(args.source)}")
    print(f"Writing MOT results to {out_path}")
    if args.video_output:
        print(f"Writing video to {args.video_output} at {fps:g} FPS")
    start_time = time.time()
    processed_frames = 0
    try:
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

                annotated = None
                if vis_dir or args.video_output:
                    annotated = draw_tracks(image, tracks, show_missing=args.show_missing_tracks)
                if vis_dir:
                    annotated.save(vis_dir / frame_path.name)
                if args.video_output:
                    video_writer.write(annotated)
                processed_frames = idx

                if idx == 1 or idx % args.log_step == 0 or idx == len(frames):
                    print(
                        f"Frame {idx}/{len(frames)} "
                        f"detections={len(detections)} active_tracks={len(tracks)} written={len(rows)}",
                        flush=True,
                    )
    finally:
        video_writer.close()
    elapsed = max(time.time() - start_time, 1e-9)
    print(f"Tracking time: {elapsed:.2f}s, FPS: {processed_frames / elapsed:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RT-DETR + hybrid Kalman/xLSTM/LNN MOT tracking.")
    parser.add_argument("--source", required=True, help="MOT sequence directory, img1 directory, or image directory")
    parser.add_argument("-c", "--config", default="configs/rtdetr/rtdetr_r50vd_6x_mot17.yml")
    parser.add_argument("-r", "--resume", required=True, help="RT-DETR detector checkpoint")
    parser.add_argument("--output", default="output/tracks.txt")
    parser.add_argument("--vis-dir", default="", help="optional directory for annotated frames")
    parser.add_argument("--video-output", default="", help="optional annotated MP4 output path")
    parser.add_argument("--video-fps", type=float, default=0.0, help="video FPS; 0 reads seqinfo.ini or uses 30")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--tracker", choices=["byte_xlstm", "hybrid"], default="byte_xlstm")
    parser.add_argument("--det-score", type=float, default=0.10)
    parser.add_argument("--track-score", type=float, default=0.45)
    parser.add_argument("--new-track-score", type=float, default=None)
    parser.add_argument("--low-track-score", type=float, default=0.10)
    parser.add_argument("--nms-iou", type=float, default=0.60)
    parser.add_argument("--person-label", type=int, default=0)
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--max-age", type=int, default=30)
    parser.add_argument("--min-hits", type=int, default=3)
    parser.add_argument("--lambda-motion", type=float, default=0.15)
    parser.add_argument("--lambda-iou", type=float, default=0.65)
    parser.add_argument("--lambda-app", type=float, default=0.20)
    parser.add_argument("--duplicate-iou", type=float, default=0.85)
    parser.add_argument("--match-cost-threshold", type=float, default=0.85)
    parser.add_argument("--low-match-cost-threshold", type=float, default=0.70)
    parser.add_argument("--motion-checkpoint", default="", help="optional trained xLSTM/LNN residual checkpoint")
    parser.add_argument("--motion-backend", choices=["xlstm", "lstm"], default="xlstm")
    parser.add_argument("--enable-neural-motion", action="store_true")
    parser.add_argument("--disable-neural-motion", action="store_true")
    parser.add_argument("--disable-color-embedding", action="store_true")
    parser.add_argument("--disable-fuse-score", action="store_true")
    parser.add_argument("--write-missing", action="store_true", help="also write predicted boxes for missing tracks")
    parser.add_argument("--show-missing-tracks", action="store_true", help="draw predicted missing tracks in video/frames")
    parser.add_argument("--log-step", type=int, default=50)
    main(parser.parse_args())
