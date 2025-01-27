import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_sequence(dataset):
    """
    Selects 5 random samples from `dataset`, which should return:
        sample['image'] -> shape (T, 4, H, W),
    and plots T=0..5 as columns in a 5-row figure (one row per sample).

    Assumptions:
      - T >= 6 so that we can visualize 6 time steps.
      - The first 3 channels in sample['image'] are B02, B03, B04 (RGB).
    """
    if len(dataset) < 5:
        raise ValueError(
            "Dataset must have at least 5 samples to plot 5 random patches."
        )

    # Randomly pick 5 distinct indices
    random_indices = random.sample(range(len(dataset)), 5)

    fig, axs = plt.subplots(nrows=5, ncols=6, figsize=(15, 10))

    for row, idx in enumerate(random_indices):
        sample = dataset[idx]
        # shape => (T, 4, H, W)
        x = sample["image"].numpy()
        print(x.shape)

        # For each time step from 0 to 5, plot the RGB channels
        for col in range(6):
            rgb = x[col, :, :, :]  # channels 0..2 => B02,B03,B04
            # Simple min-max normalization for better visualization
            vmin, vmax = rgb.min(), rgb.max()
            rgb_norm = (rgb - vmin) / (vmax - vmin + 1e-6)

            # Convert to (H, W, 3) for imshow
            rgb_norm = np.transpose(rgb_norm, (1, 2, 0))

            axs[row, col].imshow(rgb_norm)
            axs[row, col].axis("off")

            if row == 0:
                axs[row, col].set_title(f"T={col}", fontsize=12)

    plt.tight_layout()
    plt.savefig("sequence.png")


def save_animation_data(model, dataset, save_file, number=1000, device=None):
    """
    Iterate over the dataset, use the model to generate predictions,
    and save input RGB images, target NDVI, and predicted NDVI for each sample.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_rgbs, target_ndvis, pred_ndvis = [], [], []

    for idx in range(len(dataset)):
        sample = dataset[idx]
        x = sample["image"].unsqueeze(0).to(device)  # (1, T, 4, H, W)
        y = sample["target_ndvi"].unsqueeze(0).to(device)  # (1, 1, H, W)

        with torch.no_grad():
            pred = model(x)  # (1,1,H,W)

        # Process input RGB from the last time step (B02, B03, B04 => indices 0,1,2)
        # x[0, -1, 0:3, ...] => (3, H, W)
        input_rgb = x[0, -1, 0:3].cpu().numpy()  # shape (3, H, W)
        input_rgb = np.transpose(input_rgb, (1, 2, 0))  # (H, W, 3)
        # Normalize for visualization
        input_rgb = (input_rgb - input_rgb.min()) / (
            input_rgb.max() - input_rgb.min() + 1e-6
        )

        target_ndvi = y[0, 0].cpu().numpy()  # (H, W)
        pred_ndvi = pred[0, 0].cpu().numpy()

        input_rgbs.append(input_rgb)
        target_ndvis.append(target_ndvi)
        pred_ndvis.append(pred_ndvi)

        if idx >= number:
            break

    np.savez_compressed(
        save_file,
        input_rgbs=input_rgbs,
        target_ndvis=target_ndvis,
        pred_ndvis=pred_ndvis,
    )
    print(f"Animation data saved to '{save_file}'")
