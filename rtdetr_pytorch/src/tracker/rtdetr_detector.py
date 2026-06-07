import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from src.core import YAMLConfig

from .box_ops import xyxy_to_cxcywh
from .hybrid_tracker import Detection


class _DeployModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        return self.postprocessor(self.model(images), orig_target_sizes)


@dataclass
class RTDETRDetector:
    config: str
    checkpoint: str
    device: str = "cuda"
    image_size: int = 640
    score_threshold: float = 0.35
    person_label: int | None = 0
    amp: bool = True

    def __post_init__(self):
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
        self.device = torch.device(self.device)
        cfg = YAMLConfig(self.config, resume=self.checkpoint)
        checkpoint = torch.load(self.checkpoint, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint
        cfg.model.load_state_dict(state, strict=False)
        self.net = _DeployModel(cfg).to(self.device).eval()
        self.transform = T.Compose([T.Resize((self.image_size, self.image_size)), T.ToTensor()])

    @torch.no_grad()
    def __call__(self, image):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image)).convert("RGB")
        else:
            image = image.convert("RGB")

        width, height = image.size
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        orig_size = torch.tensor([[width, height]], dtype=torch.float32, device=self.device)
        use_amp = self.amp and self.device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            labels, boxes, scores = self.net(tensor, orig_size)

        labels = labels[0].detach().cpu().numpy()
        boxes = boxes[0].detach().cpu().numpy()
        scores = scores[0].detach().cpu().numpy()

        detections = []
        for label, xyxy, score in zip(labels, boxes, scores):
            if score < self.score_threshold:
                continue
            label = int(label)
            if self.person_label is not None and label != self.person_label:
                continue
            detections.append(
                Detection(
                    bbox=xyxy_to_cxcywh(xyxy),
                    score=float(score),
                    label=label,
                )
            )
        return detections


def resolve_checkpoint(path):
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path
