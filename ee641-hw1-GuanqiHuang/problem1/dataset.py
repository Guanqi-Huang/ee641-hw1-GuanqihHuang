import json
import os
from typing import Dict, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class ShapeDetectionDataset(Dataset):
    
    def __init__(self, image_dir: str, annotation_file: str, transform=None):
      self.image_dir = image_dir
      self.transform = transform

      with open(annotation_file, "r") as f:
            coco = json.load(f)

        # id -> absolute image path
      self.id_to_path: Dict[int, str] = {
            im["id"]: os.path.join(image_dir, im["file_name"]) for im in coco["images"]
        }
      self.ids = sorted(self.id_to_path.keys())  # deterministic order

      ann_by_img: Dict[int, list] = {i: [] for i in self.ids}
      for a in coco["annotations"]:
          ann_by_img.setdefault(a["image_id"], []).append(a)
      self.ann_by_img = ann_by_img

    def __len__(self): return len(self.ids)

    @staticmethod
    def _pil_to_chw_tensor(img: Image.Image) -> torch.Tensor:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:

      img_id = self.ids[idx]
      img = Image.open(self.id_to_path[img_id])
      image = self._pil_to_chw_tensor(img)  # [3,H,W] in [0,1]
      H, W = image.shape[1], image.shape[2]

      anns = self.ann_by_img.get(img_id, [])
      boxes, labels = [], []
      for a in anns:
          x, y, w, h = a["bbox"]
          boxes.append([x, y, x + w, y + h])         # xyxy
          labels.append(int(a["category_id"]))       # 0..2

      if boxes:
          boxes = torch.tensor(boxes, dtype=torch.float32)
          labels = torch.tensor(labels, dtype=torch.long)
      else:
          boxes = torch.zeros((0, 4), dtype=torch.float32)
          labels = torch.zeros((0,), dtype=torch.long)
      if self.transform is not None:
          image = self.transform(image)

      targets = {"boxes": boxes, "labels": labels, "size": torch.tensor([H, W])}
      return image, targets

def collate_fn(batch):
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)
