from collections import deque
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .box_ops import clip_cxcywh, cosine_distance, cxcywh_to_xywh, iou_cxcywh
from .motion import HybridResidualMotion, KalmanBoxFilter


@dataclass
class Detection:
    bbox: np.ndarray
    score: float
    label: int = 0
    embedding: np.ndarray | None = None


@dataclass
class Track:
    track_id: int
    bbox: np.ndarray
    score: float
    motion: KalmanBoxFilter
    history: deque
    neural_state: dict | None = None
    appearance: np.ndarray | None = None
    age: int = 1
    hits: int = 1
    missing_count: int = 0
    is_missing: bool = False
    confirmed: bool = False
    uncertainty: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))

    def mot_row(self, frame_id):
        x, y, w, h = cxcywh_to_xywh(self.bbox)
        return [frame_id, self.track_id, x, y, w, h, self.score, -1, -1, -1]


class HybridMOTTracker:
    def __init__(
        self,
        image_size,
        device="cpu",
        max_age=30,
        min_hits=3,
        score_threshold=0.35,
        low_score_threshold=0.10,
        lambda_motion=0.15,
        lambda_iou=0.65,
        lambda_app=0.20,
        max_mahalanobis=80.0,
        min_iou=0.02,
        second_stage_min_iou=0.001,
        new_track_iou_threshold=0.45,
        duplicate_iou_threshold=0.85,
        match_cost_threshold=0.85,
        low_match_cost_threshold=0.70,
        fuse_score=True,
        use_neural_motion=False,
        motion_backend="xlstm",
        motion_checkpoint=None,
    ):
        self.width, self.height = image_size
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.score_threshold = float(score_threshold)
        self.low_score_threshold = float(low_score_threshold)
        self.lambda_motion = float(lambda_motion)
        self.lambda_iou = float(lambda_iou)
        self.lambda_app = float(lambda_app)
        self.max_mahalanobis = float(max_mahalanobis)
        self.min_iou = float(min_iou)
        self.second_stage_min_iou = float(second_stage_min_iou)
        self.new_track_iou_threshold = float(new_track_iou_threshold)
        self.duplicate_iou_threshold = float(duplicate_iou_threshold)
        self.match_cost_threshold = float(match_cost_threshold)
        self.low_match_cost_threshold = float(low_match_cost_threshold)
        self.fuse_score = bool(fuse_score)
        self.tracks = []
        self.next_id = 1
        self.frame_id = 0

        self.motion_net = HybridResidualMotion(motion_backend=motion_backend).to(self.device) if use_neural_motion else None
        if self.motion_net is not None and motion_checkpoint:
            state = torch.load(motion_checkpoint, map_location=self.device)
            if "model" in state:
                state = state["model"]
            self.motion_net.load_state_dict(state, strict=False)
        if self.motion_net is not None:
            self.motion_net.eval()

    def update(self, detections, dt=1.0):
        self.frame_id += 1
        detections = [d for d in detections if d.score >= self.low_score_threshold]
        high_det_indices = [i for i, d in enumerate(detections) if d.score >= self.score_threshold]
        low_det_indices = [i for i, d in enumerate(detections) if d.score < self.score_threshold]
        predictions = [self._predict_track(track, dt) for track in self.tracks]
        all_track_indices = list(range(len(self.tracks)))
        matches, unmatched_tracks, unmatched_high_detections = self._match(
            predictions,
            detections,
            all_track_indices,
            high_det_indices,
            min_iou=self.min_iou,
            max_mahalanobis=self.max_mahalanobis,
            cost_threshold=self.match_cost_threshold,
        )

        low_matches, unmatched_tracks, _ = self._match(
            predictions,
            detections,
            unmatched_tracks,
            low_det_indices,
            min_iou=self.second_stage_min_iou,
            max_mahalanobis=self.max_mahalanobis * 2.0,
            lambda_motion=0.05,
            lambda_iou=0.90,
            lambda_app=0.05,
            cost_threshold=self.low_match_cost_threshold,
        )
        matches.extend(low_matches)

        for track_idx, det_idx in matches:
            self._update_matched(self.tracks[track_idx], detections[det_idx], dt)

        for track_idx in unmatched_tracks:
            self._update_missing(self.tracks[track_idx], predictions[track_idx], dt)

        for det_idx in unmatched_high_detections:
            if self._can_start_track(detections[det_idx]):
                self._start_track(detections[det_idx])

        self.tracks = [t for t in self.tracks if t.missing_count <= self.max_age]
        self.tracks = self._remove_duplicate_tracks(self.tracks)
        return [t for t in self.tracks if t.confirmed]

    def _predict_track(self, track, dt):
        bbox_kf, p_kf = track.motion.predict(dt)
        residual = np.zeros(4, dtype=np.float32)
        residual_unc = np.zeros(4, dtype=np.float32)

        if self.motion_net is not None and track.neural_state is not None:
            with torch.no_grad():
                history = self._history_tensor(track)
                lnn_input = self._lnn_input(track, bbox_kf, dt, is_missing=track.is_missing)
                residual, residual_unc, track.neural_state = self.motion_net.predict(
                    history,
                    lnn_input,
                    track.neural_state,
                    dt=dt,
                    missing_count=track.missing_count,
                )

        bbox_pred = clip_cxcywh(bbox_kf + residual, self.width, self.height)
        uncertainty = p_kf + np.diag(residual_unc + 1e-3).astype(np.float32)
        return bbox_pred, uncertainty

    def _match(
        self,
        predictions,
        detections,
        track_indices,
        detection_indices,
        min_iou,
        max_mahalanobis,
        lambda_motion=None,
        lambda_iou=None,
        lambda_app=None,
        cost_threshold=None,
    ):
        if not track_indices:
            return [], [], list(detection_indices)
        if not detection_indices:
            return [], list(track_indices), []

        lambda_motion = self.lambda_motion if lambda_motion is None else float(lambda_motion)
        lambda_iou = self.lambda_iou if lambda_iou is None else float(lambda_iou)
        lambda_app = self.lambda_app if lambda_app is None else float(lambda_app)
        cost_threshold = self.match_cost_threshold if cost_threshold is None else float(cost_threshold)

        cost = np.full((len(track_indices), len(detection_indices)), 1e6, dtype=np.float32)
        for local_i, track_idx in enumerate(track_indices):
            bbox_pred, uncertainty = predictions[track_idx]
            for local_j, det_idx in enumerate(detection_indices):
                det = detections[det_idx]
                motion_cost = self._mahalanobis(bbox_pred, det.bbox, uncertainty)
                iou = iou_cxcywh(bbox_pred, det.bbox)
                app_cost = cosine_distance(self.tracks[track_idx].appearance, det.embedding)
                if motion_cost > max_mahalanobis:
                    continue
                if iou < min_iou and motion_cost > max_mahalanobis * 0.35:
                    continue
                normalized_motion = min(motion_cost / max(max_mahalanobis, 1e-6), 1.0)
                cost[local_i, local_j] = (
                    lambda_motion * normalized_motion
                    + lambda_iou * (1.0 - iou)
                    + lambda_app * app_cost
                )
                if self.fuse_score:
                    score = max(float(det.score), 1e-3)
                    cost[local_i, local_j] = 1.0 - (1.0 - cost[local_i, local_j]) * score

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        used_tracks = set()
        used_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] >= 1e6 or cost[r, c] > cost_threshold:
                continue
            track_idx = int(track_indices[r])
            det_idx = int(detection_indices[c])
            matches.append((track_idx, det_idx))
            used_tracks.add(track_idx)
            used_dets.add(det_idx)

        unmatched_tracks = [i for i in track_indices if i not in used_tracks]
        unmatched_detections = [j for j in detection_indices if j not in used_dets]
        return matches, unmatched_tracks, unmatched_detections

    def _update_matched(self, track, detection, dt):
        track.motion.update(detection.bbox)
        velocity = track.motion.velocity
        feature = self._feature_vector(detection.bbox, velocity, dt, 0, 0, detection.score)
        track.history.append(feature)
        if self.motion_net is not None:
            track.neural_state = self._update_neural_state(track, detection.bbox, dt, False, 0, detection.score)
        track.bbox = detection.bbox
        track.score = detection.score
        track.age += 1
        track.hits += 1
        track.missing_count = 0
        track.is_missing = False
        track.confirmed = track.confirmed or track.hits >= self.min_hits
        if detection.embedding is not None:
            if track.appearance is None:
                track.appearance = detection.embedding
            else:
                track.appearance = 0.9 * track.appearance + 0.1 * detection.embedding

    def _update_missing(self, track, prediction, dt):
        bbox_pred, uncertainty = prediction
        velocity = track.motion.velocity
        feature = self._feature_vector(bbox_pred, velocity, dt, 1, track.missing_count + 1, 0.0)
        track.history.append(feature)
        if self.motion_net is not None:
            track.neural_state = self._update_neural_state(
                track,
                bbox_pred,
                dt,
                True,
                track.missing_count + 1,
                0.0,
            )
        track.bbox = bbox_pred
        track.score = max(0.0, track.score * 0.95)
        track.age += 1
        track.missing_count += 1
        track.is_missing = True
        track.uncertainty = uncertainty

    def _start_track(self, detection):
        kalman = KalmanBoxFilter(detection.bbox)
        history = deque(maxlen=16)
        history.append(self._feature_vector(detection.bbox, kalman.velocity, 1.0, 0, 0, detection.score))
        neural_state = self.motion_net.initial_state(self.device) if self.motion_net is not None else None
        track = Track(
            track_id=self.next_id,
            bbox=detection.bbox,
            score=detection.score,
            motion=kalman,
            history=history,
            neural_state=neural_state,
            appearance=detection.embedding,
            confirmed=self.min_hits <= 1,
        )
        self.next_id += 1
        self.tracks.append(track)

    def _can_start_track(self, detection):
        for track in self.tracks:
            if track.missing_count > 0:
                continue
            if iou_cxcywh(track.bbox, detection.bbox) >= self.new_track_iou_threshold:
                return False
        return True

    def _remove_duplicate_tracks(self, tracks):
        if len(tracks) <= 1:
            return tracks

        keep = [True] * len(tracks)
        for i in range(len(tracks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(tracks)):
                if not keep[j]:
                    continue
                if iou_cxcywh(tracks[i].bbox, tracks[j].bbox) < self.duplicate_iou_threshold:
                    continue
                drop = self._duplicate_drop_index(tracks, i, j)
                keep[drop] = False
        return [track for track, should_keep in zip(tracks, keep) if should_keep]

    @staticmethod
    def _duplicate_drop_index(tracks, i, j):
        a = tracks[i]
        b = tracks[j]
        key_a = (a.confirmed, -a.missing_count, a.hits, a.age, a.score)
        key_b = (b.confirmed, -b.missing_count, b.hits, b.age, b.score)
        return j if key_a >= key_b else i

    def _history_tensor(self, track):
        items = list(track.history)
        while len(items) < 2:
            items.insert(0, items[0])
        return torch.as_tensor(np.asarray(items, dtype=np.float32), device=self.device).unsqueeze(0)

    def _lnn_input(self, track, bbox, dt, is_missing):
        feature = self._feature_vector(
            bbox,
            track.motion.velocity,
            dt,
            int(is_missing),
            track.missing_count,
            track.score,
        )
        acceleration = np.zeros(4, dtype=np.float32)
        return torch.tensor(np.concatenate([feature, acceleration]), dtype=torch.float32, device=self.device).unsqueeze(0)

    def _update_neural_state(self, track, bbox, dt, is_missing, missing_count, score):
        feature = self._feature_vector(
            bbox,
            track.motion.velocity,
            dt,
            int(is_missing),
            missing_count,
            score,
        )
        acceleration = np.zeros(4, dtype=np.float32)
        history = self._history_tensor(track)
        lnn_input = torch.tensor(
            np.concatenate([feature, acceleration]),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            return self.motion_net.update_state(history, lnn_input, track.neural_state, dt)

    def _feature_vector(self, bbox, velocity, dt, is_missing, missing_count, score):
        scale = np.array([self.width, self.height, self.width, self.height], dtype=np.float32)
        bbox_norm = np.asarray(bbox, dtype=np.float32) / np.maximum(scale, 1.0)
        vel_norm = np.asarray(velocity[:4], dtype=np.float32) / np.maximum(scale, 1.0)
        return np.concatenate(
            [
                bbox_norm,
                vel_norm,
                np.array([float(dt), float(is_missing), float(missing_count), float(score)], dtype=np.float32),
            ]
        )

    @staticmethod
    def _mahalanobis(pred, det, uncertainty):
        diff = np.asarray(det, dtype=np.float32) - np.asarray(pred, dtype=np.float32)
        cov = uncertainty + np.eye(4, dtype=np.float32) * 1e-3
        return float(diff.T @ np.linalg.inv(cov) @ diff)
