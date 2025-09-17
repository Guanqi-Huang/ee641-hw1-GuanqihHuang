import os
import json
from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def _try_load_annotations(anno_path: str):
    with open(anno_path, "r") as f:
        data = json.load(f)
    if not (isinstance(data, dict) and "images" in data):
        raise ValueError("Expected {'images': [...]} at %s" % anno_path)

    samples = []
    for im in data["images"]:
        samples.append({
            "file_name": im["file_name"],       # e.g., "000000.png"
            "keypoints": im["keypoints"],       # list of K [x,y] pairs
        })
    return samples


class KeypointDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        annotation_file: str,
        output_type: str = "heatmap",
        heatmap_size: int = 64,
        sigma: float = 2.0,
        num_keypoints: int = 5,
    ) -> None:
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.output_type = output_type
        self.heatmap_size = int(heatmap_size)
        self.sigma = float(sigma)
        self.num_keypoints = int(num_keypoints)

        self.samples = _try_load_annotations(annotation_file)
        if len(self.samples) == 0:
            raise RuntimeError("No samples found in annotations: %s" % annotation_file)

        # Validate shapes
        for s in self.samples:
            kp = s["keypoints"]
            if not (isinstance(kp, list) and len(kp) == self.num_keypoints and
                    all(isinstance(p, (list, tuple)) and len(p) == 2 for p in kp)):
                raise ValueError(
                    f"Expected keypoints as {self.num_keypoints} pairs [[x,y],...], got {kp} in {self.annotation_file}"
                )


    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _to_tensor_gray(img: Image.Image) -> torch.Tensor:
        if img.mode != "L":
            img = img.convert("L")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # [H,W]
        t = torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]
        return t

    @staticmethod
    def _gaussian_2d(height: int, width: int, cx: float, cy: float, sigma: float) -> np.ndarray:
        ys = np.arange(height, dtype=np.float32)[:, None]
        xs = np.arange(width, dtype=np.float32)[None, :]
        return np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma * sigma))

    def generate_heatmap(self, keypoints_xy: np.ndarray, height: int, width: int) -> torch.Tensor:
        assert keypoints_xy.shape == (self.num_keypoints, 2)
        scale_x = width / 128.0
        scale_y = height / 128.0
        H = np.zeros((self.num_keypoints, height, width), dtype=np.float32)
        for i, (x, y) in enumerate(keypoints_xy):
            cx = x * scale_x
            cy = y * scale_y
            # Clamp center inside map
            cx = np.clip(cx, 0, width - 1)
            cy = np.clip(cy, 0, height - 1)
            H[i] = self._gaussian_2d(height, width, cx, cy, self.sigma)
        return torch.from_numpy(H)

    def __getitem__(self, idx: int):
        rec = self.samples[idx]
        img_path = os.path.join(self.image_dir, rec["file_name"])
        img = Image.open(img_path)
        # Ensure training images are 128x128
        if img.size != (128, 128):
            img = img.resize((128, 128), Image.BILINEAR)

        image = self._to_tensor_gray(img)  # [1,128,128]

        kps = np.array(rec["keypoints"], dtype=np.float32).reshape(self.num_keypoints, 2)
        meta = {
            "file_name": rec["file_name"],
            "orig_w": 128,
            "orig_h": 128,
            "keypoints": kps.copy(),  # pixel coords
        }

        if self.output_type == "heatmap":
            target = self.generate_heatmap(kps, self.heatmap_size, self.heatmap_size)  # [K,Hm,Wm]
        elif self.output_type == "regression":
            # Normalize to [0,1] by dividing by image size
            target = torch.from_numpy((kps / 128.0).reshape(-1))  # [2K]
        else:
            raise ValueError("output_type must be 'heatmap' or 'regression'")

        return image.float(), target.float(), meta