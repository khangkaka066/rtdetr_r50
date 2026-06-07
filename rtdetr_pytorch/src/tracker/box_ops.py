import numpy as np


def xyxy_to_cxcywh(box):
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return np.array(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5, max(0.0, x2 - x1), max(0.0, y2 - y1)],
        dtype=np.float32,
    )


def cxcywh_to_xyxy(box):
    cx, cy, w, h = np.asarray(box, dtype=np.float32)
    return np.array(
        [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5],
        dtype=np.float32,
    )


def cxcywh_to_xywh(box):
    cx, cy, w, h = np.asarray(box, dtype=np.float32)
    return np.array([cx - w * 0.5, cy - h * 0.5, w, h], dtype=np.float32)


def clip_cxcywh(box, width, height):
    xyxy = cxcywh_to_xyxy(box)
    xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, width)
    xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, height)
    return xyxy_to_cxcywh(xyxy)


def iou_cxcywh(a, b):
    a = cxcywh_to_xyxy(a)
    b = cxcywh_to_xyxy(b)
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def cosine_distance(a, b):
    if a is None or b is None:
        return 1.0
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / denom)
