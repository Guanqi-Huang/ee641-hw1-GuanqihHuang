import os
import json
import argparse
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def train_heatmap_model(model, train_loader, val_loader, device, out_dir, num_epochs=30):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val = float("inf")
    log = {"heatmap": {"train_loss": [], "val_loss": []}}
    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        train_loss = running / len(train_loader.dataset)
        model.eval()
        running = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                loss = criterion(preds, targets)
                running += loss.item() * images.size(0)
        val_loss = running / len(val_loader.dataset)
        log["heatmap"]["train_loss"].append(train_loss)
        log["heatmap"]["val_loss"].append(val_loss)
        print(f"[Heatmap][{epoch:03d}] train {train_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "heatmap_model.pth"))
    return log


def train_regression_model(model, train_loader, val_loader, device, out_dir, num_epochs=30):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val = float("inf")
    log = {"regression": {"train_loss": [], "val_loss": []}}
    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        for images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            preds = model(images)
            loss = criterion(preds, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                preds = model(images)
                loss = criterion(preds, targets)
                running += loss.item() * images.size(0)
        val_loss = running / len(val_loader.dataset)

        log["regression"]["train_loss"].append(train_loss)
        log["regression"]["val_loss"].append(val_loss)
        print(f"[Regress][{epoch:03d}] train {train_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "regression_model.pth"))

    return log


def make_loaders(train_images, train_ann, val_images, val_ann, batch_size, heatmap_size=64, sigma=2.0):
    train_heat = KeypointDataset(train_images, train_ann, output_type="heatmap",
                                 heatmap_size=heatmap_size, sigma=sigma)
    val_heat = KeypointDataset(val_images, val_ann, output_type="heatmap",
                               heatmap_size=heatmap_size, sigma=sigma)

    train_reg = KeypointDataset(train_images, train_ann, output_type="regression")
    val_reg = KeypointDataset(val_images, val_ann, output_type="regression")

    heat_train_loader = DataLoader(train_heat, batch_size=batch_size, shuffle=True, num_workers=0)
    heat_val_loader = DataLoader(val_heat, batch_size=batch_size, shuffle=False, num_workers=0)

    reg_train_loader = DataLoader(train_reg, batch_size=batch_size, shuffle=True, num_workers=0)
    reg_val_loader = DataLoader(val_reg, batch_size=batch_size, shuffle=False, num_workers=0)

    return heat_train_loader, heat_val_loader, reg_train_loader, reg_val_loader


def main():
    parser = argparse.ArgumentParser(description="EE641 HW1 Problem 2 Training")
    parser.add_argument("--train_images", type=str, required=True, help="Path to training image directory")
    parser.add_argument("--train_ann", type=str, required=True, help="Path to training annotation JSON")
    parser.add_argument("--val_images", type=str, required=True, help="Path to validation image directory")
    parser.add_argument("--val_ann", type=str, required=True, help="Path to validation annotation JSON")
    parser.add_argument("--out_dir", type=str, default="results", help="Output directory for models/logs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--heatmap_size", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "visualizations"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    heat_train_loader, heat_val_loader, reg_train_loader, reg_val_loader = make_loaders(
        args.train_images, args.train_ann, args.val_images, args.val_ann,
        batch_size=args.batch_size, heatmap_size=args.heatmap_size, sigma=args.sigma
    )

    # Heatmap model
    heatmap_model = HeatmapNet(num_keypoints=5).to(device)
    log_h = train_heatmap_model(heatmap_model, heat_train_loader, heat_val_loader, device,
                                args.out_dir, num_epochs=args.epochs)

    # Regression model
    regression_model = RegressionNet(num_keypoints=5).to(device)
    log_r = train_regression_model(regression_model, reg_train_loader, reg_val_loader, device,
                                   args.out_dir, num_epochs=args.epochs)

    # Merge logs and save
    training_log = {}
    training_log.update(log_h)
    training_log.update(log_r)
    with open(os.path.join(args.out_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)


if __name__ == "__main__":
    main()