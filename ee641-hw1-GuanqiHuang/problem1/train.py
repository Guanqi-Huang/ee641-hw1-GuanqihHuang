import os,
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from dataset import ShapeDetectionDataset, collate_fn
from model   import MultiScaleDetector
from loss    import DetectionLoss
from utils   import generate_anchors

PROBLEM1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBLEM1_DIR.parent  
DATA_ROOT    = PROJECT_ROOT / "datasets" / "detection"

TRAIN_IMG_DIR = str(DATA_ROOT / "train")
TRAIN_ANN     = str(DATA_ROOT / "train_annotations.json")
VAL_IMG_DIR   = str(DATA_ROOT / "val")
VAL_ANN       = str(DATA_ROOT / "val_annotations.json")

RESULTS_DIR = PROBLEM1_DIR / "results"
VIZ_DIR     = RESULTS_DIR / "visualizations"
CKPT_PATH   = RESULTS_DIR / "best_model.pth"
LOG_PATH    = RESULTS_DIR / "training_log.json"

BATCH_SIZE = 16
EPOCHS     = 50
LR         = 1e-3
MOMENTUM   = 0.9

def _build_anchors():
    fm_sizes = [(56, 56), (28, 28), (14, 14)]
    scales   = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    return generate_anchors(fm_sizes, scales, image_size=224)

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ShapeDetectionDataset(TRAIN_IMG_DIR, TRAIN_ANN)
    val_ds   = ShapeDetectionDataset(VAL_IMG_DIR,   VAL_ANN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)

    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    criterion = DetectionLoss(num_classes=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

    anchors = [a.to(device) for a in _build_anchors()]

    best_val = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        #train
        model.train()
        tr = {"loss":0., "obj":0., "cls":0., "loc":0.}
        for imgs, targets in train_dl:
            imgs = imgs.to(device)
            preds = model(imgs)
            d = criterion(preds, targets, anchors)
            loss = d["loss_total"]
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            tr["loss"] += float(loss.item()); tr["obj"] += float(d["loss_obj"].item())
            tr["cls"]  += float(d["loss_cls"].item()); tr["loc"] += float(d["loss_loc"].item())
        n = max(1, len(train_dl))
        for k in tr: tr[k] /= n

        # validate
        model.eval()
        va = {"loss":0., "obj":0., "cls":0., "loc":0.}
        with torch.no_grad():
            for imgs, targets in val_dl:
                imgs = imgs.to(device)
                preds = model(imgs)
                d = criterion(preds, targets, anchors)
                va["loss"] += float(d["loss_total"].item()); va["obj"] += float(d["loss_obj"].item())
                va["cls"]  += float(d["loss_cls"].item());  va["loc"] += float(d["loss_loc"].item())
        m = max(1, len(val_dl))
        for k in va: va[k] /= m
        history.append({"epoch": epoch, "train": tr, "val": va})
        print(f"[{epoch:03d}] train {tr} | val {va}")
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(model.state_dict(), CKPT_PATH)
        with open(LOG_PATH, "w") as f:
            json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
