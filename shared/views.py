"""Decode-once fixed ROI views shared by teacher and student evaluation."""
from contextlib import closing

from PIL import Image
import torch
from torch.utils.data import Dataset

from .augmentation import build_image_transforms
from .config import AugmentationConfig
from .image_io import load_rgb_image
from .roi import fixed_roi_views


class FixedViewsDataset(Dataset):
    def __init__(self, source, boxes, data, augmentation=None, *, scales=None, shift_ratio=None):
        if boxes is not None and len(boxes) != len(source.paths):
            raise ValueError("Fixed-view boxes must match every source row.")
        self.source = source
        self.boxes = boxes if data.crop_bbox else None
        self.data = data
        self.scales = data.roi_scales if scales is None else scales
        self.shift_ratio = data.roi_shift_ratio if shift_ratio is None else shift_ratio
        self.transform = build_image_transforms(data, augmentation or AugmentationConfig()).evaluation

    def __len__(self):
        return len(self.source.paths)

    def __getitem__(self, index):
        path = self.source.paths[index]
        box = self.boxes[index] if self.boxes is not None else None
        values = []
        try:
            image, _ = load_rgb_image(path, allow_truncated=self.data.allow_truncated_images)
            with closing(image):
                for view in fixed_roi_views(image, box, size=self.data.image_size,
                                            method=self.data.crop_method, scales=self.scales,
                                            shift_ratio=self.shift_ratio):
                    with closing(view):
                        values.append(self.transform(view))
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise RuntimeError(f"Failed to load {self.source.name}[{index}]: {path}") from error
        return torch.stack(values), index
