from typing import List, Tuple
import torch

# Anchor generation
def generate_anchors(feature_map_sizes: List[Tuple[int, int]],
                     anchor_scales: List[List[float]],
                     image_size: int = 224) -> List[torch.Tensor]:
    all_levels = []
    for (H, W), scales in zip(feature_map_sizes, anchor_scales):
        A = len(scales)
        stride_y = image_size / H
        stride_x = image_size / W
        anchors = torch.zeros((H * W * A, 4), dtype=torch.float32)
        k = 0
        for i in range(H):
            cy = (i + 0.5) * stride_y
            for j in range(W):
                cx = (j + 0.5) * stride_x
                for s in scales:
                    w = float(s); h = float(s)
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                    anchors[k] = torch.tensor([x1, y1, x2, y2])
                    k += 1
        all_levels.append(anchors)
    return all_levels


#Box operation
def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = x2 - x1
    h = y2 - y1
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return torch.stack([cx, cy, w, h], dim=-1)

def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def compute_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    #Compute IoU between two sets of boxes.
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0] if boxes2.dim() else 0), dtype=torch.float32)

    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    inter_x1 = torch.max(x11[:, None], x21[None, :])
    inter_y1 = torch.max(y11[:, None], y21[None, :])
    inter_x2 = torch.min(x12[:, None], x22[None, :])
    inter_y2 = torch.min(y12[:, None], y22[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h
    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter
    iou = torch.where(union > 0, inter / union, torch.zeros_like(union))
    return iou

# Encode and Decode
def encode(gt_boxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """
    Encode GT boxes relative to anchors using (t_x, t_y, t_w, t_h):
    t_x = (cx_gt - cx_a) / w_a,  t_y = (cy_gt - cy_a) / h_a,
    t_w = log(w_gt / w_a),       t_h = log(h_gt / h_a).
    All inputs are xyxy; outputs are [N,4].
    """
    g = _xyxy_to_cxcywh(gt_boxes)
    a = _xyxy_to_cxcywh(anchors)
    tx = (g[:, 0] - a[:, 0]) / (a[:, 2].clamp(min=1e-6))
    ty = (g[:, 1] - a[:, 1]) / (a[:, 3].clamp(min=1e-6))
    tw = torch.log((g[:, 2].clamp(min=1e-6)) / (a[:, 2].clamp(min=1e-6)))
    th = torch.log((g[:, 3].clamp(min=1e-6)) / (a[:, 3].clamp(min=1e-6)))
    return torch.stack([tx, ty, tw, th], dim=-1)

def decode(deltas: torch.Tensor, anchors: torch.Tensor, image_size: int = 224) -> torch.Tensor:
    a = _xyxy_to_cxcywh(anchors)
    cx = deltas[:, 0] * a[:, 2] + a[:, 0]
    cy = deltas[:, 1] * a[:, 3] + a[:, 1]
    w  = torch.exp(deltas[:, 2]) * a[:, 2]
    h  = torch.exp(deltas[:, 3]) * a[:, 3]
    boxes = _cxcywh_to_xyxy(torch.stack([cx, cy, w, h], dim=-1))
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_size - 1)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_size - 1)
    return boxes

#Anchor and GT matching
def match_anchors_to_targets(
    anchors: torch.Tensor,
    target_boxes: torch.Tensor,
    target_labels: torch.Tensor,
    pos_threshold: float = 0.5,
    neg_threshold: float = 0.3,
):
    device = anchors.device
    N = anchors.shape[0]
    C = int(target_labels.max().item() + 1) if target_labels.numel() > 0 else 0

    matched_labels = torch.zeros((N,), dtype=torch.long, device=device)
    matched_boxes  = torch.zeros((N, 4), dtype=torch.float32, device=device)
    pos_mask = torch.zeros((N,), dtype=torch.bool, device=device)
    neg_mask = torch.zeros((N,), dtype=torch.bool, device=device)

    if target_boxes.numel() == 0:
        neg_mask[:] = True
        return matched_labels, matched_boxes, pos_mask, neg_mask

    iou = compute_iou(anchors, target_boxes)  # [N,T]
    best_iou, gt_idx = iou.max(dim=1) 
    pos_mask = best_iou >= pos_threshold
    neg_mask = best_iou <  neg_threshold
    best_anchor_per_gt = iou.argmax(dim=0)   # [T]
    pos_mask[best_anchor_per_gt] = True
    gt_idx[best_anchor_per_gt]   = torch.arange(target_boxes.size(0), device=device)

    neg_mask = neg_mask & (~pos_mask)

    if pos_mask.any():
        sel = gt_idx[pos_mask]
        matched_boxes[pos_mask]  = target_boxes[sel]
        # labels in training are shifted by +1 for positives
        matched_labels[pos_mask] = target_labels[sel] + 1

    return matched_labels, matched_boxes, pos_mask, neg_mask
