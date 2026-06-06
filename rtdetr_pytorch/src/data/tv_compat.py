"""Torchvision v2 tensor compatibility helpers.

Older RT-DETR code used ``torchvision.datapoints``. Newer torchvision versions
renamed these wrappers to ``torchvision.tv_tensors`` and changed
``BoundingBox`` to ``BoundingBoxes`` with ``canvas_size`` instead of
``spatial_size``.
"""

from __future__ import annotations

import torchvision


if hasattr(torchvision, "disable_beta_transforms_warning"):
    torchvision.disable_beta_transforms_warning()


try:
    from torchvision import datapoints as _tv

    IMAGE_TYPE = _tv.Image
    VIDEO_TYPE = _tv.Video
    MASK_TYPE = _tv.Mask
    BOUNDING_BOX_TYPE = _tv.BoundingBox
    BOUNDING_BOX_FORMAT = _tv.BoundingBoxFormat

    def make_bounding_box(data, format, spatial_size):
        return _tv.BoundingBox(data, format=format, spatial_size=spatial_size)

    def make_mask(data):
        return _tv.Mask(data)

    def get_spatial_size(box):
        return box.spatial_size

except ImportError:
    from torchvision import tv_tensors as _tv

    IMAGE_TYPE = _tv.Image
    VIDEO_TYPE = _tv.Video
    MASK_TYPE = _tv.Mask
    BOUNDING_BOX_TYPE = _tv.BoundingBoxes
    BOUNDING_BOX_FORMAT = _tv.BoundingBoxFormat

    def make_bounding_box(data, format, spatial_size):
        return _tv.BoundingBoxes(data, format=format, canvas_size=tuple(spatial_size))

    def make_mask(data):
        return _tv.Mask(data)

    def get_spatial_size(box):
        return box.canvas_size
