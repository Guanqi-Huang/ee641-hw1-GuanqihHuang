import os
import sys
import math
from pathlib import Path
from typing import List, Dict, Tuple
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(__file__))
from dataset import ShapeDetectionDataset, collate_fn
from model   import MultiScaleDetector
from utils   import generate_anchors, decode, compute_iou

PROBLEM1_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROBLEM1_DIR.parent
DATA_ROOT    = PROJECT_ROOT / "datasets" / "detection"
VAL_IMG_DIR  = str(DATA_ROOT / "val")
VAL_ANN      = str(DATA_ROOT / "val_annotations.json")
RESULTS_DIR  = PROBLEM1_DIR / "results"
VIZ_DIR      = RESULTS_DIR / "visualizations"
CKPT_PATH    = RESULTS_DIR / "best_model.pth"

IMAGE_SIZE   = 224
FEATURE_MAP_SIZES = [(56, 56), (28, 28), (14, 14)]
ANCHOR_SCALES     = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
NUM_CLASSES  = 3
NUM_ANCHORS  = 3

# Inference thresholds
SCORE_THR        = 0.20           
NMS_IOU_THR      = 0.50
TOPK_PER_CLASS   = 200 
MAX_DETECTIONS   = 300 

#Numerically stable sigmoid used for objectness score prediction
def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

#Create multi-scale square anchors
def _build_anchors() -> List[torch.Tensor]: 
    return generate_anchors(FEATURE_MAP_SIZES, ANCHOR_SCALES, image_size=IMAGE_SIZE)

# Perform per-class Non-Maximum Suppression (NMS).
def _nms_per_class(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float, topk: int) -> torch.Tensor:
    if boxes.numel() == 0 or scores.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)
    order = scores.argsort(descending=True)[:topk]
    keep: List[int] = []
    while order.numel() > 0:
        i = int(order[0].item())
        keep.append(i)
        if order.numel() == 1:
            break
        ious = compute_iou(boxes[i].unsqueeze(0), boxes[order[1:]])[0]  # [M]
        remaining = (ious <= iou_thr).nonzero(as_tuple=False).squeeze(1)
        order = order[1:][remaining]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)

# Convert raw model outputs into final detections and returns list of dictionary per image.
def predict(
    preds: List[torch.Tensor],
    anchors: List[torch.Tensor],
    score_thresh: float = SCORE_THR,
    iou_thr: float = NMS_IOU_THR,
    topk_per_class: int = TOPK_PER_CLASS,
    max_dets: int = MAX_DETECTIONS,
) -> List[Dict[str, torch.Tensor]]:
    B = preds[0].shape[0]
    num_classes = preds[0].shape[1] // NUM_ANCHORS - 5
    out: List[Dict[str, torch.Tensor]] = []

    for b in range(B):
        # Step1: collect all decoded anchors across scales
        level_boxes, level_scores, level_labels = [], [], []
        for s, P in enumerate(preds):
            H, W = P.shape[-2:]
            A = NUM_ANCHORS
            X = P[b].permute(1, 2, 0).reshape(H * W * A, 5 + num_classes)  # [N,5+C]
            loc, obj, cls = X[:, :4], _sigmoid(X[:, 4]), F.softmax(X[:, 5:], dim=-1)  # [N,4],[N],[N,C]
            boxes = decode(loc, anchors[s].to(P.device))  # [N,4] xyxy
            # prefilter by score threshold
            cls_scores, cls_labels = cls.max(dim=-1)     # [N], [N]
            scores = cls_scores * obj
            keep = scores >= score_thresh
            if keep.any():
                level_boxes.append(boxes[keep])
                level_scores.append(scores[keep])
                level_labels.append(cls_labels[keep])

        if not level_boxes:
            out.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            })
            continue
        boxes  = torch.cat(level_boxes, dim=0)
        scores = torch.cat(level_scores, dim=0)
        labels = torch.cat(level_labels, dim=0)
        final_boxes, final_scores, final_labels = [], [], []
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                kb = boxes[mask]
                ks = scores[mask]
                keep_idx = _nms_per_class(kb, ks, iou_thr=iou_thr, topk=topk_per_class)
                if keep_idx.numel() > 0:
                    final_boxes.append(kb[keep_idx])
                    final_scores.append(ks[keep_idx])
                    final_labels.append(torch.full((keep_idx.numel(),), c, dtype=torch.long, device=kb.device))

        if final_boxes:
            fb = torch.cat(final_boxes,  dim=0)
            fs = torch.cat(final_scores, dim=0)
            fl = torch.cat(final_labels, dim=0)

            if fb.shape[0] > max_dets:
                o = fs.argsort(descending=True)[:max_dets]
                fb, fs, fl = fb[o], fs[o], fl[o]

            out.append({"boxes": fb.detach().cpu(),
                        "scores": fs.detach().cpu(),
                        "labels": fl.detach().cpu()})
        else:
            out.append({
                "boxes":  torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.long),
            })
    return out

def compute_ap(preds: List[Dict[str, torch.Tensor]],
               gts:   List[Dict[str, torch.Tensor]],
               iou_thr: float = 0.5) -> Tuple[Dict[int, float], float]:
    """Compute Average Precision for a single class."""
    assert len(preds) == len(gts)
    C = 3
    aps: Dict[int, float] = {}
    for c in range(C):
        # Collect all predictions of class c across images
        flat: List[Tuple[float, int, torch.Tensor]] = []  # (score, img_idx, box[1,4])
        total_gt = 0
        for i, (p, g) in enumerate(zip(preds, gts)):
            # count GTs of this class
            if g["labels"].numel() > 0:
                total_gt += int((g["labels"] == c).sum().item())
            # add predicted boxes of this class
            mask = (p["labels"] == c)
            if mask.any():
                for j in torch.nonzero(mask, as_tuple=False).squeeze(1).tolist():
                    flat.append((float(p["scores"][j].item()), i, p["boxes"][j].unsqueeze(0)))

        # Sort predictions by confidence
        flat.sort(key=lambda t: t[0], reverse=True)

        tp, fp = [], []
        matched = [set() for _ in gts]  # which gt indices are already matched per image
        for _, i, pb in flat:
            if gts[i]["boxes"].numel() == 0:
                fp.append(1); tp.append(0); continue
            ious = compute_iou(pb, gts[i]["boxes"])[0]  # [num_gt]
            j = int(torch.argmax(ious).item())
            if ious[j] >= iou_thr and j not in matched[i] and int(gts[i]["labels"][j]) == c:
                tp.append(1); fp.append(0); matched[i].add(j)
            else:
                tp.append(0); fp.append(1)

        if total_gt == 0 or len(tp) == 0:
            aps[c] = 0.0
            continue

        tps = torch.tensor(tp, dtype=torch.float32)
        fps = torch.tensor(fp, dtype=torch.float32)
        cum_tp = torch.cumsum(tps, dim=0)
        cum_fp = torch.cumsum(fps, dim=0)
        recall    = cum_tp / total_gt
        precision = cum_tp / torch.clamp(cum_tp + cum_fp, min=1.0)

        # area under PR curve (sorted by recall)
        r, idx = torch.sort(recall)
        p = precision[idx]
        aps[c] = float(torch.trapz(p, r).item())

    mAP = (sum(aps.values()) / max(1, len(aps))) if aps else 0.0
    return aps, mAP

def _draw(img_path: str, pred: Dict[str, torch.Tensor], gt: Dict[str, torch.Tensor], save_path: str):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    # draw GT (green)
    for i in range(gt["boxes"].shape[0]):
        x1, y1, x2, y2 = [int(v) for v in gt["boxes"][i].tolist()]
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=2)
    # draw predictions (red) with score
    for i in range(pred["boxes"].shape[0]):
        x1, y1, x2, y2 = [int(v) for v in pred["boxes"][i].tolist()]
        score = float(pred["scores"][i].item())
        draw.rectangle([x1, y1, x2, y2], outline=(255,0,0), width=2)
        draw.text((x1+2, y1+2), f"{score:.2f}", fill=(255,0,0))
    img.save(save_path)

def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    ds = ShapeDetectionDataset(VAL_IMG_DIR, VAL_ANN)
    dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=True, collate_fn=collate_fn)

    # Model + anchors
    model = MultiScaleDetector(num_classes=NUM_CLASSES, num_anchors=NUM_ANCHORS).to(device)
    if CKPT_PATH.exists():
        model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()
    anchors = [a.to(device) for a in _build_anchors()]

    preds_all: List[Dict[str, torch.Tensor]] = []
    gts_all:   List[Dict[str, torch.Tensor]] = []

    with torch.no_grad():
        for imgs, targets in dl:
            imgs = imgs.to(device)
            outputs = model(imgs)
            batch_preds = predict(outputs, anchors,
                                  score_thresh=SCORE_THR, iou_thr=NMS_IOU_THR,
                                  topk_per_class=TOPK_PER_CLASS, max_dets=MAX_DETECTIONS)
            preds_all.extend(batch_preds)
            # collect GT
            for t in targets:
                gts_all.append({
                    "boxes":  t["boxes"],
                    "labels": t["labels"],
                })

    # Visualize first few validation images
    vis_count = 0
    for i in range(min(24, len(ds))):
        img_id   = ds.ids[i]
        img_path = ds.id_to_path[img_id]
        _draw(img_path, preds_all[i], gts_all[i], str(VIZ_DIR / f"val_{i:02d}.png"))
        vis_count += 1

    # Metrics
    aps, mAP = compute_ap(preds_all, gts_all, iou_thr=0.5)
    with open(VIZ_DIR / "metrics.txt", "w") as f:
        f.write(f"mAP@0.5: {mAP:.4f}\nAP per class: {aps}\n")

    print(f"mAP@0.5={mAP:.4f}  AP per class={aps}")
    print(f"Saved {vis_count} images to: {VIZ_DIR}")

if __name__ == "__main__":
    main()

def compute_ap(predictions, ground_truths, iou_threshold=0.5):
    """
    Compute Average Precision (AP) at a single IoU threshold for all classes.
    Returns: (aps: Dict[class_id, AP], mAP: float)
    """
    # Reuse the implementation already present under the same name if exists.
    return _compute_ap_impl(predictions, ground_truths, iou_threshold)

def visualize_detections(image, predictions, ground_truths, save_path):
    #Draw predicted boxes (red with score) and GT boxes (green) and save to disk.
    if isinstance(image, str):
        img_path = image
        img = Image.open(image).convert("RGB")
    else:
        img = image
        img_path = None
    _draw_impl(img, predictions, ground_truths, save_path)

def analyze_scale_performance(model, dataloader, anchors):
    #Generate simple stats showing which scale fires on which box sizes.
    model.eval()
    stats = {0: {0:0, 1:0, 2:0}, 1: {0:0, 1:0, 2:0}, 2: {0:0, 1:0, 2:0}}
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(next(model.parameters()).device)
            outputs = model(imgs)  # list of 3 levels
            # For each level, decode + quick threshold and count by class
            for s, P in enumerate(outputs):
                H, W = P.shape[-2:]
                A = P.shape[1] // (5 + 3)
                X = P.permute(0,2,3,1).reshape(imgs.size(0), H*W*A, 5+3)
                loc = X[..., :4]
                obj = torch.sigmoid(X[..., 4])
                cls = torch.softmax(X[..., 5:], dim=-1)
                for b in range(imgs.size(0)):
                    boxes_s = decode(loc[b].reshape(-1,4), anchors[s].to(imgs.device))
                    scores, labels = cls[b].max(-1)
                    final = scores * obj[b]
                    keep = final >= 0.2
                    if keep.any():
                        for c in range(3):
                            cnt = int(((labels[keep] == c).sum()).item())
                            stats[s][c] += cnt
    return stats
