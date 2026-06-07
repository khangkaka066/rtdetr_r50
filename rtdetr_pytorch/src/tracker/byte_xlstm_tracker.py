from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .box_ops import cosine_distance, cxcywh_to_xywh, iou_cxcywh
from .hybrid_tracker import Detection
from .motion import HybridResidualMotion


class TrackState(Enum):
    Tracked = 1
    Lost = 2
    Removed = 3


class KalmanXYAH:
    """ByteTrack/DeepSORT-style Kalman filter over [x, y, aspect, height]."""

    def __init__(self):
        ndim = 4
        dt = 1.0
        self.motion_mat = np.eye(2 * ndim, dtype=np.float32)
        for i in range(ndim):
            self.motion_mat[i, ndim + i] = dt
        self.update_mat = np.eye(ndim, 2 * ndim, dtype=np.float32)
        self.std_weight_position = 1.0 / 20.0
        self.std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros_like(measurement)].astype(np.float32)
        std = [
            2 * self.std_weight_position * measurement[3],
            2 * self.std_weight_position * measurement[3],
            1e-2,
            2 * self.std_weight_position * measurement[3],
            10 * self.std_weight_velocity * measurement[3],
            10 * self.std_weight_velocity * measurement[3],
            1e-5,
            10 * self.std_weight_velocity * measurement[3],
        ]
        covariance = np.diag(np.square(std)).astype(np.float32)
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-2,
            self.std_weight_position * mean[3],
        ]
        std_vel = [
            self.std_weight_velocity * mean[3],
            self.std_weight_velocity * mean[3],
            1e-5,
            self.std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov
        return mean.astype(np.float32), covariance.astype(np.float32)

    def project(self, mean, covariance):
        std = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std)).astype(np.float32)
        mean = self.update_mat @ mean
        covariance = self.update_mat @ covariance @ self.update_mat.T
        return mean.astype(np.float32), (covariance + innovation_cov).astype(np.float32)

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        kalman_gain = covariance @ self.update_mat.T @ np.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        mean = mean + kalman_gain @ innovation
        covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.T
        return mean.astype(np.float32), covariance.astype(np.float32)


def cxcywh_to_xyah(bbox):
    cx, cy, w, h = np.asarray(bbox, dtype=np.float32)
    h = max(float(h), 1.0)
    return np.array([cx, cy, w / h, h], dtype=np.float32)


def xyah_to_cxcywh(xyah):
    cx, cy, aspect, h = np.asarray(xyah, dtype=np.float32)
    h = max(float(h), 1.0)
    w = max(float(aspect) * h, 1.0)
    return np.array([cx, cy, w, h], dtype=np.float32)


@dataclass
class ByteTrackXLSTMTrack:
    detection: Detection
    track_id: int = 0
    mean: np.ndarray | None = None
    covariance: np.ndarray | None = None
    state: TrackState = TrackState.Tracked
    is_activated: bool = False
    frame_id: int = 0
    start_frame: int = 0
    tracklet_len: int = 0
    hits: int = 0
    age: int = 0
    missing_count: int = 0
    history: deque | None = None
    neural_state: dict | None = None
    appearance: np.ndarray | None = None

    @property
    def score(self):
        return float(self.detection.score)

    @score.setter
    def score(self, value):
        self.detection.score = float(value)

    @property
    def bbox(self):
        if self.mean is None:
            return self.detection.bbox.copy()
        return xyah_to_cxcywh(self.mean[:4])

    @property
    def is_missing(self):
        return self.state == TrackState.Lost

    @property
    def confirmed(self):
        return self.is_activated and self.state == TrackState.Tracked

    def activate(self, kalman, track_id, frame_id, neural_state=None, feature=None):
        self.track_id = track_id
        self.mean, self.covariance = kalman.initiate(cxcywh_to_xyah(self.detection.bbox))
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0
        self.hits = 1
        self.age = 1
        self.missing_count = 0
        self.neural_state = neural_state
        self.history = deque(maxlen=16)
        if feature is not None:
            self.history.append(feature)
        self.appearance = self.detection.embedding

    def predict(self, kalman):
        if self.mean is None or self.covariance is None:
            return
        mean = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean[7] = 0.0
        self.mean, self.covariance = kalman.predict(mean, self.covariance)

    def update(self, kalman, detection, frame_id, feature=None, neural_state=None):
        self.mean, self.covariance = kalman.update(
            self.mean,
            self.covariance,
            cxcywh_to_xyah(detection.bbox),
        )
        self.detection = detection
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.hits += 1
        self.age += 1
        self.missing_count = 0
        if feature is not None:
            self.history.append(feature)
        if neural_state is not None:
            self.neural_state = neural_state
        if detection.embedding is not None:
            if self.appearance is None:
                self.appearance = detection.embedding
            else:
                self.appearance = 0.9 * self.appearance + 0.1 * detection.embedding

    def mark_lost(self):
        self.state = TrackState.Lost
        self.missing_count += 1

    def mark_removed(self):
        self.state = TrackState.Removed

    def mot_row(self, frame_id):
        x, y, w, h = cxcywh_to_xywh(self.bbox)
        return [frame_id, self.track_id, x, y, w, h, self.score, -1, -1, -1]


class ByteTrackXLSTMTracker:
    def __init__(
        self,
        image_size,
        device="cpu",
        track_thresh=0.65,
        low_thresh=0.12,
        new_track_thresh=None,
        match_thresh=0.85,
        second_match_thresh=0.70,
        unconfirmed_match_thresh=0.70,
        track_buffer=45,
        min_hits=1,
        nms_duplicate_iou=0.75,
        lambda_iou=0.80,
        lambda_app=0.12,
        fuse_score=True,
        use_xlstm=False,
        motion_backend="xlstm",
        motion_checkpoint=None,
    ):
        self.width, self.height = image_size
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.track_thresh = float(track_thresh)
        self.low_thresh = float(low_thresh)
        self.new_track_thresh = float(new_track_thresh if new_track_thresh is not None else track_thresh + 0.1)
        self.match_thresh = float(match_thresh)
        self.second_match_thresh = float(second_match_thresh)
        self.unconfirmed_match_thresh = float(unconfirmed_match_thresh)
        self.max_time_lost = int(track_buffer)
        self.min_hits = int(min_hits)
        self.nms_duplicate_iou = float(nms_duplicate_iou)
        self.lambda_iou = float(lambda_iou)
        self.lambda_app = float(lambda_app)
        self.fuse_score = bool(fuse_score)
        self.frame_id = 0
        self.next_id = 1
        self.kalman = KalmanXYAH()
        self.tracked_tracks = []
        self.lost_tracks = []
        self.removed_tracks = []

        self.motion_net = HybridResidualMotion(motion_backend=motion_backend).to(self.device) if use_xlstm else None
        if self.motion_net is not None and motion_checkpoint:
            state = torch.load(motion_checkpoint, map_location=self.device)
            if "model" in state:
                state = state["model"]
            self.motion_net.load_state_dict(state, strict=False)
        if self.motion_net is not None:
            self.motion_net.eval()

    def update(self, detections, dt=1.0):
        self.frame_id += 1
        detections = [d for d in detections if d.score >= self.low_thresh]
        high = [self._new_track(d) for d in detections if d.score >= self.track_thresh]
        low = [self._new_track(d) for d in detections if self.low_thresh <= d.score < self.track_thresh]

        activated = []
        refind = []
        lost = []
        removed = []

        unconfirmed = [t for t in self.tracked_tracks if not t.is_activated]
        tracked = [t for t in self.tracked_tracks if t.is_activated]
        track_pool = self._joint_tracks(tracked, self.lost_tracks)
        self._predict_tracks(track_pool, dt)

        matches, u_track, u_det = self._associate(track_pool, high, self.match_thresh)
        for track_idx, det_idx in matches:
            track = track_pool[track_idx]
            det = high[det_idx]
            self._update_existing(track, det, dt)
            if track.state == TrackState.Tracked:
                activated.append(track)
            else:
                refind.append(track)

        remaining_tracked = [track_pool[i] for i in u_track if track_pool[i].state == TrackState.Tracked]
        low_matches, u_remaining, _ = self._associate(remaining_tracked, low, self.second_match_thresh)
        for track_idx, det_idx in low_matches:
            track = remaining_tracked[track_idx]
            det = low[det_idx]
            self._update_existing(track, det, dt)
            activated.append(track)

        for idx in u_remaining:
            track = remaining_tracked[idx]
            track.mark_lost()
            lost.append(track)

        remaining_high = [high[i] for i in u_det]
        unconfirmed_matches, u_unconfirmed, u_det = self._associate(
            unconfirmed,
            remaining_high,
            self.unconfirmed_match_thresh,
        )
        for track_idx, det_idx in unconfirmed_matches:
            track = unconfirmed[track_idx]
            det = remaining_high[det_idx]
            self._update_existing(track, det, dt)
            activated.append(track)
        for idx in u_unconfirmed:
            unconfirmed[idx].mark_removed()
            removed.append(unconfirmed[idx])

        for idx in u_det:
            track = remaining_high[idx]
            if track.score < self.new_track_thresh:
                continue
            self._activate_new(track, dt)
            activated.append(track)

        for track in self.lost_tracks:
            if self.frame_id - track.frame_id > self.max_time_lost:
                track.mark_removed()
                removed.append(track)

        self.tracked_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        self.tracked_tracks = self._joint_tracks(self.tracked_tracks, activated)
        self.tracked_tracks = self._joint_tracks(self.tracked_tracks, refind)
        self.lost_tracks = self._sub_tracks(self.lost_tracks, self.tracked_tracks)
        self.lost_tracks = self._joint_tracks(self.lost_tracks, lost)
        self.lost_tracks = self._sub_tracks(self.lost_tracks, removed)
        self.removed_tracks = self._joint_tracks(self.removed_tracks, removed)
        self.tracked_tracks, self.lost_tracks = self._remove_duplicate_tracks(
            self.tracked_tracks,
            self.lost_tracks,
        )

        return [t for t in self.tracked_tracks if t.is_activated and t.hits >= self.min_hits]

    def _new_track(self, detection):
        return ByteTrackXLSTMTrack(detection=detection)

    def _activate_new(self, track, dt):
        feature = self._feature_vector(track.detection.bbox, np.zeros(4, dtype=np.float32), dt, 0, 0, track.score)
        neural_state = self.motion_net.initial_state(self.device) if self.motion_net is not None else None
        track.activate(self.kalman, self.next_id, self.frame_id, neural_state=neural_state, feature=feature)
        self.next_id += 1

    def _update_existing(self, track, new_track, dt):
        feature = self._feature_vector(new_track.detection.bbox, self._velocity_from_track(track), dt, 0, 0, new_track.score)
        neural_state = self._update_neural_state(track, new_track.detection.bbox, dt, 0, 0, new_track.score)
        track.update(self.kalman, new_track.detection, self.frame_id, feature=feature, neural_state=neural_state)

    def _predict_tracks(self, tracks, dt):
        for track in tracks:
            track.predict(self.kalman)
            if self.motion_net is None or track.neural_state is None:
                continue
            with torch.no_grad():
                bbox_kf = track.bbox
                history = self._history_tensor(track)
                lnn_input = self._lnn_input(track, bbox_kf, dt, is_missing=track.is_missing)
                residual, _, track.neural_state = self.motion_net.predict(
                    history,
                    lnn_input,
                    track.neural_state,
                    dt=dt,
                    missing_count=track.missing_count,
                )
            corrected_bbox = bbox_kf + residual
            track.mean[:4] = cxcywh_to_xyah(corrected_bbox)

    def _associate(self, tracks, detections, thresh):
        if not tracks:
            return [], [], list(range(len(detections)))
        if not detections:
            return [], list(range(len(tracks))), []

        cost = np.full((len(tracks), len(detections)), 1e6, dtype=np.float32)
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou = iou_cxcywh(track.bbox, det.detection.bbox)
                app_cost = cosine_distance(track.appearance, det.detection.embedding)
                base_cost = self.lambda_iou * (1.0 - iou) + self.lambda_app * app_cost
                if self.fuse_score:
                    base_cost = 1.0 - (1.0 - base_cost) * max(det.score, 1e-3)
                cost[i, j] = base_cost

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        used_tracks = set()
        used_dets = set()
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] > thresh:
                continue
            matches.append((int(r), int(c)))
            used_tracks.add(int(r))
            used_dets.add(int(c))
        u_tracks = [i for i in range(len(tracks)) if i not in used_tracks]
        u_dets = [i for i in range(len(detections)) if i not in used_dets]
        return matches, u_tracks, u_dets

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

    def _history_tensor(self, track):
        items = list(track.history or [])
        if not items:
            items = [self._feature_vector(track.bbox, self._velocity_from_track(track), 1.0, 0, 0, track.score)]
        while len(items) < 2:
            items.insert(0, items[0])
        return torch.as_tensor(np.asarray(items, dtype=np.float32), device=self.device).unsqueeze(0)

    def _lnn_input(self, track, bbox, dt, is_missing):
        feature = self._feature_vector(
            bbox,
            self._velocity_from_track(track),
            dt,
            int(is_missing),
            track.missing_count,
            track.score,
        )
        acceleration = np.zeros(4, dtype=np.float32)
        return torch.as_tensor(np.concatenate([feature, acceleration]), dtype=torch.float32, device=self.device).unsqueeze(0)

    def _update_neural_state(self, track, bbox, dt, is_missing, missing_count, score):
        if self.motion_net is None or track.neural_state is None:
            return None
        feature = self._feature_vector(bbox, self._velocity_from_track(track), dt, is_missing, missing_count, score)
        history_items = list(track.history or [])
        history_items.append(feature)
        history = torch.as_tensor(np.asarray(history_items, dtype=np.float32), device=self.device).unsqueeze(0)
        acceleration = np.zeros(4, dtype=np.float32)
        lnn_input = torch.as_tensor(np.concatenate([feature, acceleration]), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.motion_net.update_state(history, lnn_input, track.neural_state, dt)

    @staticmethod
    def _velocity_from_track(track):
        if track.mean is None:
            return np.zeros(4, dtype=np.float32)
        h = max(float(track.mean[3]), 1.0)
        return np.array([track.mean[4], track.mean[5], track.mean[6] * h, track.mean[7]], dtype=np.float32)

    @staticmethod
    def _joint_tracks(a, b):
        exists = set()
        result = []
        for track in a + b:
            if track.track_id in exists:
                continue
            exists.add(track.track_id)
            result.append(track)
        return result

    @staticmethod
    def _sub_tracks(a, b):
        remove_ids = {track.track_id for track in b}
        return [track for track in a if track.track_id not in remove_ids]

    def _remove_duplicate_tracks(self, tracked, lost):
        tracked_remove = set()
        lost_remove = set()
        for i, track_a in enumerate(tracked):
            for j, track_b in enumerate(lost):
                if iou_cxcywh(track_a.bbox, track_b.bbox) < self.nms_duplicate_iou:
                    continue
                life_a = track_a.frame_id - track_a.start_frame
                life_b = track_b.frame_id - track_b.start_frame
                if life_a >= life_b:
                    lost_remove.add(j)
                else:
                    tracked_remove.add(i)
        tracked = [t for i, t in enumerate(tracked) if i not in tracked_remove]
        lost = [t for i, t in enumerate(lost) if i not in lost_remove]
        return tracked, lost
