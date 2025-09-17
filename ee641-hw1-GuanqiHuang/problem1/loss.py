from typing import Dict, List
import torch
import torch.nn as nn
from utils import match_anchors_to_targets, encode

# Hard Negative Mining (negatives : positives)
HNM_RATIO = 3

# Loss weights
W_OBJ, W_CLS, W_LOC = 1.0, 1.0, 2.0

class DetectionLoss(nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.C = num_classes
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.ce  = nn.CrossEntropyLoss(reduction="none")
        self.l1  = nn.SmoothL1Loss(reduction="none")

    @torch.no_grad()
    def hard_negative_mining(
        self,
        obj_loss_all: torch.Tensor,   # [N] BCE loss for every anchor on a level for one image
        pos_mask: torch.Tensor,       # [N] bool
        neg_mask: torch.Tensor,       # [N] bool
        ratio: int = HNM_RATIO,
    ) -> torch.Tensor:
        """
        Select the hardest negatives so that #neg ~= ratio * #pos (up to availability).
        Returns a boolean mask the same shape as neg_mask indicating selected negatives.
        """
        pos = int(pos_mask.sum().item())
        avail = int(neg_mask.sum().item())

       
        if pos == 0:
            k = min(100, avail)
            if k == 0:
                return neg_mask & (obj_loss_all == obj_loss_all)
            thr = torch.topk(obj_loss_all[neg_mask], k=k, largest=True).values.min()
            return neg_mask & (obj_loss_all >= thr)
        k = min(ratio * pos, avail)
        if k == 0:
            return neg_mask & (obj_loss_all == obj_loss_all)
        thr = torch.topk(obj_loss_all[neg_mask], k=k, largest=True).values.min()
        return neg_mask & (obj_loss_all >= thr)

    def forward(
        self,
        predictions: List[torch.Tensor],                # list of S tensors: [B, (5+C)*A, H, W]
        targets: List[Dict[str, torch.Tensor]],         # len B dicts: {"boxes":[Ti,4], "labels":[Ti]}
        anchors: List[torch.Tensor],                    # list of S tensors: [H*W*A, 4] (xyxy)
    ) -> Dict[str, torch.Tensor]:
        # Accumulators
        device = predictions[0].device
        zero   = predictions[0].new_tensor(0.0)
        loss_obj, loss_cls, loss_loc = zero.clone(), zero.clone(), zero.clone()

        B = predictions[0].shape[0]

        for b in range(B):
            yb = targets[b]["boxes"].to(device)    # [Tb, 4] xyxy
            yl = targets[b]["labels"].to(device)   # [Tb]

            for s, P_all in enumerate(predictions):
                # Shape bookkeeping
                H, W = P_all.shape[-2:]
                A = anchors[s].size(0) // (H * W)  # anchors per cell at this level
                P = P_all[b].permute(1, 2, 0).reshape(H * W * A, 5 + self.C)
                loc, obj_logit, cls_logit = P[:, :4], P[:, 4], P[:, 5:]

                anc = anchors[s].to(device)        # [N, 4]

                labels, mboxes, pos_mask, neg_mask = match_anchors_to_targets(
                    anc, yb, yl
                )
                obj_t = torch.zeros_like(obj_logit)
                obj_t[pos_mask] = 1.0

                obj_all = self.bce(obj_logit, obj_t)  # [N]
                sel_neg = self.hard_negative_mining(obj_all.detach(), pos_mask, neg_mask)
                mask    = pos_mask | sel_neg

                if mask.any():
                    loss_obj += obj_all[mask].mean()
                else:
                    loss_obj += zero

                if pos_mask.any():
                    # Shift labels to {0..C-1} for CE
                    cls_t = (labels[pos_mask] - 1).clamp(min=0)
                    loss_cls += self.ce(cls_logit[pos_mask], cls_t).mean()

                    # Encode GT w.r.t. anchors and regress with SmoothL1
                    t = encode(mboxes[pos_mask], anc[pos_mask])
                    loss_loc += self.l1(loc[pos_mask], t).mean()

        # Average over batch
        N = max(1, B)
        loss_obj = loss_obj / N
        loss_cls = loss_cls / N
        loss_loc = loss_loc / N

        return {
            "loss_obj":   loss_obj,
            "loss_cls":   loss_cls,
            "loss_loc":   loss_loc,
            "loss_total": W_OBJ * loss_obj + W_CLS * loss_cls + W_LOC * loss_loc,
        }
