import os
import json
import argparse
from typing import List, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

def extract_keypoints_from_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Returns pixel-space coords [B, K, 2] for a 128x128 input.
    """
    B, K, H, W = heatmaps.shape
    with torch.no_grad():
        flat = heatmaps.view(B, K, -1)
        idx = flat.argmax(dim=-1)      # [B, K]
        ys = (idx // W).float()        # [B, K]
        xs = (idx % W).float()         # [B, K]
        xs = xs * (128.0 / float(W))
        ys = ys * (128.0 / float(H))
        coords = torch.stack([xs, ys], dim=-1)  # [B, K, 2]
    return coords


def compute_pck(predictions: np.ndarray, ground_truths: np.ndarray, thresholds: List[float]) -> Dict[float, float]:
    N, K, _ = predictions.shape
    mins = ground_truths.min(axis=1)             # [N, 2]
    maxs = ground_truths.max(axis=1)             # [N, 2]
    diags = np.linalg.norm(maxs - mins, axis=1)  # [N]

    dists = np.linalg.norm(predictions - ground_truths, axis=-1)  # [N, K]
    pck = {}
    for thr in thresholds:
        tol = diags[:, None] * thr               # [N, 1]
        correct = (dists <= tol).astype(np.float32).mean()  # over all N*K
        pck[thr] = float(correct)
    return pck


def plot_pck_curves(pck_heatmap: Dict[float, float], pck_regression: Dict[float, float], save_path: str):
    # normalize keys from either floats or strings → floats
    pck_h = {float(k): float(v) for k, v in pck_heatmap.items()}
    pck_r = {float(k): float(v) for k, v in pck_regression.items()}

    thrs = sorted(set(pck_h.keys()) | set(pck_r.keys()))
    ys_h = [pck_h.get(t, np.nan) for t in thrs]
    ys_r = [pck_r.get(t, np.nan) for t in thrs]
    plt.figure()
    plt.plot(thrs, ys_h, marker="o", label="Heatmap")
    plt.plot(thrs, ys_r, marker="s", label="Regression")
    plt.xlabel("PCK threshold (fraction of bbox diag)")
    plt.ylabel("Accuracy")
    plt.title("PCK Curves")
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def visualize_predictions(image: torch.Tensor, pred_kps: np.ndarray, gt_kps: np.ndarray, save_path: str):
    img = image.squeeze(0).cpu().numpy()
    plt.figure()
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)
    plt.scatter(gt_kps[:, 0], gt_kps[:, 1], s=30, marker="o", facecolors="none", edgecolors="g", label="GT")
    plt.scatter(pred_kps[:, 0], pred_kps[:, 1], s=20, marker="x", label="Pred")
    for i in range(gt_kps.shape[0]):
        plt.text(gt_kps[i, 0] + 2, gt_kps[i, 1] + 2, f"{i}", color="y", fontsize=8)
    plt.legend()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="EE641 HW1 Problem 2 Evaluation")
    parser.add_argument("--mode", choices=["heatmap", "regression"], required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--ann", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="results/visualizations")
    parser.add_argument("--num_samples", type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build dataset andmodel
    if args.mode == "heatmap":
        ds = KeypointDataset(args.images, args.ann, output_type="heatmap")
        net = HeatmapNet(num_keypoints=5).to(device)
    else:
        ds = KeypointDataset(args.images, args.ann, output_type="regression")
        net = RegressionNet(num_keypoints=5).to(device)

    state = torch.load(args.model_path, map_location=device)
    net.load_state_dict(state)
    net.eval()
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    all_preds: List[np.ndarray] = []
    all_gts: List[np.ndarray] = []
    saved = 0
    with torch.no_grad():
        for images, targets, meta in loader:
            images = images.to(device)
            # Predictions -> pixel coords [B,K,2]
            if args.mode == "heatmap":
                heatmaps = net(images)                               # [B,K,64,64]
                coords = extract_keypoints_from_heatmaps(heatmaps)   # [B,K,2]
            else:
                coords_norm = net(images)                            # [B,2K] in [0,1]
                B = coords_norm.shape[0]
                coords = coords_norm.view(B, -1, 2) * 128.0          # [B,K,2] pixels
            # Ground-truth keypoints [B,K,2] from meta (dict-of-lists)
            if isinstance(meta, dict) and "keypoints" in meta:
                gts = torch.as_tensor(meta["keypoints"]).float()
            else:  # fallback (rare)
                gts = torch.stack([torch.as_tensor(m["keypoints"]) for m in meta]).float()
            all_preds.append(coords.cpu().numpy())
            all_gts.append(gts.numpy())
            # Save  visualizations
            for b in range(images.size(0)):
                if saved >= args.num_samples:
                    break
                visualize_predictions(
                    images[b].cpu(),
                    coords[b].cpu().numpy(),
                    gts[b].cpu().numpy(),
                    os.path.join(args.out_dir, f"viz_{args.mode}_{saved:03d}.png"),
                )
                saved += 1
            if saved >= args.num_samples:
                break
    # Stack and score
    preds = np.concatenate(all_preds, axis=0)  # [N,K,2]
    gts   = np.concatenate(all_gts,   axis=0)  # [N,K,2]

    thresholds = [0.05, 0.10, 0.15, 0.20]
    pck = compute_pck(preds, gts, thresholds)
    print("PCK:", json.dumps(pck, indent=2))
    # Save PCK json 
    with open(os.path.join(os.path.dirname(args.out_dir), f"pck_{args.mode}.json"), "w") as f:
        json.dump(pck, f, indent=2)
    # Plot 
    other = "regression" if args.mode == "heatmap" else "heatmap"
    other_json = os.path.join(os.path.dirname(args.out_dir), f"pck_{other}.json")
    if os.path.exists(other_json):
        with open(other_json, "r") as f:
            pck_other = json.load(f)
        if args.mode == "heatmap":
            plot_pck_curves(pck, pck_other, os.path.join(args.out_dir, "pck_curves.png"))
        else:
            plot_pck_curves(pck_other, pck, os.path.join(args.out_dir, "pck_curves.png"))

if __name__ == "__main__":
    main()
