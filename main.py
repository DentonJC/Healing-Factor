#!/usr/bin/env python3

import argparse
import os

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import (MonthlyTimeSeries, PatchDataset, find_and_mosaic_s2,
                  generate_monthly_date_ranges, save_patches_for_dataset)
from model import ConvLSTM
from utils import plot_sequence, save_animation_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--minx", type=float, default=48.9, help="Min longitude of bounding box."
    )
    parser.add_argument(
        "--miny", type=float, default=38.9, help="Min latitude of bounding box."
    )
    parser.add_argument(
        "--maxx", type=float, default=49.0, help="Max longitude of bounding box."
    )
    parser.add_argument(
        "--maxy", type=float, default=39.0, help="Max latitude of bounding box."
    )
    parser.add_argument(
        "--start_date", type=str, default="2020-05-01", help="Start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end_date", type=str, default="2020-10-31", help="End date (YYYY-MM-DD)."
    )

    parser.add_argument(
        "--max_scenes_per_month",
        type=int,
        default=3,
        help="Max number of scenes per month to use for mosaicing.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="donbas_s2_data",
        help="Directory to store monthly mosaic GeoTIFFs.",
    )

    parser.add_argument(
        "--seq_len", type=int, default=3, help="Number of months to use as input."
    )
    parser.add_argument(
        "--patch_size", type=int, default=128, help="Patch size for training samples."
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")

    parser.add_argument(
        "--overwrite_patches",
        action="store_true",
        help="Overwrite patch files if they already exist.",
    )
    return parser.parse_args()


def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training"):
        x = batch["image"].to(device)
        y = batch["target_ndvi"].to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0.0


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            x = batch["image"].to(device)
            y = batch["target_ndvi"].to(device)
            pred = model(x)
            loss = criterion(pred, y)
            total_loss += loss.item()
    return total_loss / len(loader) if len(loader) > 0 else 0.0


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ROI_BOUNDS = {
        "minx": args.minx,
        "miny": args.miny,
        "maxx": args.maxx,
        "maxy": args.maxy,
    }

    os.makedirs(args.data_dir, exist_ok=True)

    # Connect to Planetary Computer STAC
    stac_url = "https://planetarycomputer.microsoft.com/api/stac/v1"
    client = pystac_client.Client.open(stac_url)

    # Generate monthly mosaics
    monthly_tifs = []
    for (yyyymm, start_d, end_d) in generate_monthly_date_ranges(
        args.start_date, args.end_date
    ):
        mosaic_path = find_and_mosaic_s2(
            yyyymm,
            start_d,
            end_d,
            ROI_BOUNDS,
            args.data_dir,
            args.max_scenes_per_month,
            stac_client=client,
        )
        if mosaic_path is not None:
            monthly_tifs.append(mosaic_path)

    if not monthly_tifs:
        raise RuntimeError(
            "No monthly mosaics were created. Check bounding box or date range."
        )
    monthly_tifs = sorted(monthly_tifs)

    # Build spatiotemporal dataset (T=seq_len => predict NDVI of month T+1)
    full_dataset = MonthlyTimeSeries(monthly_tifs, seq_len=args.seq_len)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])

    # Generate and save spatiotemporal (because of order) patches
    train_patches_dir = "train_patches"
    val_patches_dir = "val_patches"
    save_patches_for_dataset(
        train_data, args.patch_size, train_patches_dir, overwrite=args.overwrite_patches
    )
    save_patches_for_dataset(
        val_data, args.patch_size, val_patches_dir, overwrite=args.overwrite_patches
    )

    train_ds = PatchDataset(train_patches_dir)
    val_ds = PatchDataset(val_patches_dir)

    plot_sequence(val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1, drop_last=True
    )

    model = ConvLSTM(in_channels=4, hidden_channels=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_no_improve += 1

        print(
            f"[Epoch {epoch+1}/{args.epochs}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )

        if epochs_no_improve >= 10:
            print("Early stopping triggered.")
            break

    print("Training complete!")

    model.load_state_dict(torch.load("best_model.pth"))
    save_file = "animation_data.npz"
    save_animation_data(model, val_ds, save_file, device=device)


if __name__ == "__main__":
    main()
