#!/usr/bin/env python3

import os
import shutil
from datetime import datetime, timedelta
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import pystac_client
import rasterio
import requests
from rasterio.merge import merge
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


def generate_monthly_date_ranges(start_str: str, end_str: str):
    """
    Yields tuples (YYYY-MM, month_start_date, month_end_date) for each month
    in the interval [start_str, end_str].
    """
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")
    current_date = start_date.replace(day=1)

    while current_date <= end_date:
        yyyymm = current_date.strftime("%Y-%m")
        # Move to next month
        next_month = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = next_month - timedelta(days=1)
        if month_end > end_date:
            month_end = end_date
        yield (
            yyyymm,
            current_date.strftime("%Y-%m-%d"),
            month_end.strftime("%Y-%m-%d"),
        )
        current_date = next_month


def pythonic_download(url: str, local_path: str) -> None:
    """
    Downloads a file from 'url' to 'local_path' using streaming requests.
    """
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def median_merge_method(dest: np.ndarray, src: tuple) -> None:
    """
    Custom median merge function for rasterio.merge(..., method=callable).

    dest: shape (count=1, height, width)
    src : (data, mask) with data.shape => (N, height, width)
          and mask.shape => (N, height, width), nonzero=valid pixels
    """
    data, mask = src  # data, mask => (N, H, W)
    _, h, w = data.shape
    for row in range(h):
        for col in range(w):
            valid_idx = np.nonzero(mask[:, row, col])
            if len(valid_idx[0]) == 0:
                continue
            vals = data[:, row, col][valid_idx]
            if len(vals) > 0:
                dest[0, row, col] = np.median(vals)


################################################################################
# DOWNLOADING & MOSAICKING
################################################################################


def find_and_mosaic_s2(
    yyyymm: str,
    date_start: str,
    date_end: str,
    bbox: dict,
    out_dir: str,
    max_scenes: int,
    stac_client,
):
    """
    Searches Planetary Computer STAC for Sentinel-2 L2A in [date_start, date_end] & bounding box.
    Downloads up to 'max_scenes' scenes with the lowest cloud cover.
    Merges them into a pixelwise median mosaic for bands B02, B03, B04, B08.
    Returns the path to the final multi-band monthly GeoTIFF, or None if no mosaic.
    """
    out_mosaic_path = os.path.join(out_dir, f"{yyyymm}_S2_4band.tif")
    if os.path.exists(out_mosaic_path):
        print(f"Skipping {yyyymm} - mosaic already exists: {out_mosaic_path}")
        return out_mosaic_path

    print(f"\n=== [Monthly Mosaic] Processing {yyyymm} ===")

    search_bbox = [bbox["minx"], bbox["miny"], bbox["maxx"], bbox["maxy"]]
    search = stac_client.search(
        collections=["sentinel-2-l2a"],
        bbox=search_bbox,
        datetime=f"{date_start}T00:00:00Z/{date_end}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": 80}},
    )
    items = list(search.get_items())
    if len(items) == 0:
        print(f"No Sentinel-2 items found for {yyyymm}. Skipping.")
        return None

    # Sort by ascending cloud cover and pick first N
    items.sort(key=lambda it: it.properties.get("eo:cloud_cover", 100))
    items = items[:max_scenes]

    # Temporary directory
    tmp_dir = os.path.join(out_dir, f"{yyyymm}_temp")
    os.makedirs(tmp_dir, exist_ok=True)

    # For each item, download B02, B03, B04, B08 (if present)
    band_files = {"B02": [], "B03": [], "B04": [], "B08": []}
    for item in items:
        signed_item = planetary_computer.sign(item)
        item_id = item.id
        for b in band_files.keys():
            if b not in signed_item.assets:
                continue
            asset = signed_item.assets[b]
            local_path = os.path.join(tmp_dir, f"{item_id}_{b}.tif")
            if not os.path.exists(local_path):
                pythonic_download(asset.href, local_path)
            band_files[b].append(local_path)

    # Merge each band
    merged_band_paths = []
    reference_meta = None
    for band, file_list in band_files.items():
        if len(file_list) == 0:
            print(f"Missing band {band} in {yyyymm}, skipping band.")
            merged_band_paths.append(None)
            continue

        src_files = [rasterio.open(fp) for fp in file_list]
        mosaic_arr, out_trans = merge(src_files, method=median_merge_method)
        # mosaic_arr => (1, H, W)
        meta = src_files[0].meta.copy()
        meta.update(
            {
                "transform": out_trans,
                "height": mosaic_arr.shape[1],
                "width": mosaic_arr.shape[2],
                "count": mosaic_arr.shape[0],
                "compress": "deflate",
                "dtype": rasterio.float32,
            }
        )

        for s in src_files:
            s.close()

        # Save mosaic for this band
        band_mosaic_path = os.path.join(tmp_dir, f"{yyyymm}_{band}_mosaic.tif")
        with rasterio.open(band_mosaic_path, "w", **meta) as dst:
            dst.write(mosaic_arr.astype(np.float32))

        merged_band_paths.append((band, band_mosaic_path))
        if reference_meta is None:
            reference_meta = meta

    # Stack B02, B03, B04, B08 in order if they exist
    final_bands = ["B02", "B03", "B04", "B08"]
    stacked_arrays = []
    actual_bands = []
    for band_name in final_bands:
        match = [p for (b, p) in merged_band_paths if b == band_name and p is not None]
        if len(match) == 1:
            with rasterio.open(match[0]) as src:
                stacked_arrays.append(src.read(1))  # shape (H, W)
            actual_bands.append(band_name)

    if len(stacked_arrays) == 0:
        print(f"No valid bands found for {yyyymm}. Cleaning up.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # Create final mosaic
    stacked_data = np.stack(stacked_arrays, axis=0)  # shape => (num_bands, H, W)
    final_meta = reference_meta.copy()
    final_meta.update(
        {
            "count": len(actual_bands),
            "dtype": rasterio.float32,
        }
    )

    with rasterio.open(out_mosaic_path, "w", **final_meta) as dst:
        dst.write(stacked_data.astype(np.float32))

    print(f" -> Created monthly mosaic {out_mosaic_path} with bands {actual_bands}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_mosaic_path


################################################################################
# DATASET DEFINITIONS
################################################################################


class MonthlyTimeSeries(Dataset):
    """
    Each index i => {'image': (T, 4, H, W), 'target_ndvi': (1, H, W)},
    for T=seq_len monthly mosaics, predicting NDVI of the next month.
    """

    def __init__(self, tif_paths: List[str], seq_len=3):
        super().__init__()
        self.tif_paths = sorted(tif_paths)
        self.seq_len = seq_len
        # We'll form examples from months [i..i+seq_len-1] => predict i+seq_len
        self.indices = list(range(len(self.tif_paths) - self.seq_len))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_idx = self.indices[idx]
        input_paths = self.tif_paths[start_idx : start_idx + self.seq_len]
        target_path = self.tif_paths[start_idx + self.seq_len]

        # Load input stacks => shape (T, 4, H, W)
        monthly_data = []
        for p in input_paths:
            with rasterio.open(p) as src:
                arr = src.read()  # shape (4, H, W)
            monthly_data.append(arr)
        input_stack = np.stack(monthly_data, axis=0)  # (T,4,H,W)

        # Compute NDVI of the target month
        with rasterio.open(target_path) as src:
            arr_t = src.read()  # shape (4, H, W)
        red = arr_t[2]  # B04
        nir = arr_t[3]  # B08
        ndvi = (nir - red) / (nir + red + 1e-6)

        sample = {
            "image": torch.from_numpy(input_stack).float(),  # (T,4,H,W)
            "target_ndvi": torch.from_numpy(ndvi).float().unsqueeze(0),  # (1,H,W)
        }
        return sample


def save_patches_for_dataset(
    dataset: Dataset, patch_size: int, output_dir: str, overwrite=False
):
    """
    Cuts the (T,4,H,W) 'image' and (1,H,W) 'target_ndvi' from each dataset sample
    into non-overlapping patches of size patch_size, and saves each patch as .npz.
    """

    # If patches already exist and we're not overwriting, skip.
    if not overwrite and os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
        print(
            f"Patches already exist in '{output_dir}' and overwrite is disabled. Skipping."
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    patch_index = 0

    for idx in range(len(dataset)):
        sample = dataset[idx]
        x = sample["image"].numpy()  # shape (T,4,H,W)
        y = sample["target_ndvi"].numpy()  # shape (1,H,W)
        T, C, H, W = x.shape

        # Non-overlapping patch creation
        for row_start in range(0, H, patch_size):
            for col_start in range(0, W, patch_size):
                if (row_start + patch_size <= H) and (col_start + patch_size <= W):
                    x_patch = x[
                        :,
                        :,
                        row_start : row_start + patch_size,
                        col_start : col_start + patch_size,
                    ]
                    y_patch = y[
                        :,
                        row_start : row_start + patch_size,
                        col_start : col_start + patch_size,
                    ]
                    patch_path = os.path.join(output_dir, f"patch_{patch_index}.npz")
                    np.savez_compressed(patch_path, image=x_patch, target=y_patch)
                    patch_index += 1

    print(f"Saved {patch_index} patches to '{output_dir}'")


class PatchDataset(Dataset):
    """
    Loads each (T,4,patch_size,patch_size) patch and the NDVI target (1,patch_size,patch_size)
    from disk.
    """

    def __init__(self, patch_dir: str):
        super().__init__()
        self.patch_dir = patch_dir
        self.files = [
            os.path.join(patch_dir, f)
            for f in os.listdir(patch_dir)
            if f.endswith(".npz")
        ]
        self.files.sort()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data_path = self.files[idx]
        data = np.load(data_path)
        x = data["image"]  # (T,4,ph,pw)
        y = data["target"]  # (1,ph,pw)
        return {
            "image": torch.from_numpy(x).float(),
            "target_ndvi": torch.from_numpy(y).float(),
        }
