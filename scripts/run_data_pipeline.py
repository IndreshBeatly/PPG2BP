#!/usr/bin/env python3
"""
run_full_pipeline.py

For each raw .npz file in a specified folder (each must contain 'ppg','abp','fs=500'),
this script will:
  1) Strip first 30 min and last 10 min (strip_times)
  2) Remove NaN indices (nan_index_removal)
  3) Downsample from 500 Hz → 50 Hz (downsample_to_50hz)
  4) Split into non‐overlapping 10 s windows, compute SNR, and keep top 300 (ten_sec_splitting)
  5) Compute SBP (global max) and DBP (median of troughs) for each window (sbp_dbp_labels)
  6) Save a final .npz containing {ppg_windows, sbp_values, dbp_values, fs=50}

All intermediate outputs are created in a temporary directory and removed automatically.
Only the “_final.npz” remains in the output folder.

Usage:
    chmod +x run_full_pipeline.py
    ./run_full_pipeline.py --raw_dir /path/to/raw_data --out_dir /path/to/output_folder
"""

import os
import glob
import tempfile
import argparse
import numpy as np

# Import each module you converted:
from modules.strip_time import strip_times             
from modules.nan_removal import remove_nans           
from modules.decimation50hz import downsample_to_50hz         
from modules.tensec_window_split import select_top_windows_with_snr,compute_snr   
from modules.bp_labels import sbp_dbp_labels,compute_sbp_dbp            

def run_pipeline_on_single_file(raw_path: str, output_dir: str) -> None:
    """
    Runs the five‐step pipeline on one raw .npz file,
    writing only a final '<basename>_final.npz' into output_dir.
    All intermediate files live in a TemporaryDirectory and get auto‐deleted.
    """
    base = os.path.splitext(os.path.basename(raw_path))[0]
    final_name = f"{base}_final.npz"
    final_path = os.path.join(output_dir, final_name)

    # If final already exists, skip
    if os.path.exists(final_path):
        print(f"→ Skipping '{base}' (final already exists).")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1) Strip first 30 min & last 10 min
        step1 = os.path.join(tmpdir, f"{base}_step1_stripped.npz")
        strip_times(raw_path, step1)

        # 2) Remove NaN samples
        step2 = os.path.join(tmpdir, f"{base}_step2_nanclean.npz")
        remove_nans(step1, step2)

        # 3) Downsample to 50 Hz
        step3 = os.path.join(tmpdir, f"{base}_step3_downsampled.npz")
        downsample_to_50hz(step2, step3)

        # 4) Split into top‐300 windows by SNR
        step4 = os.path.join(tmpdir, f"{base}_step4_topsnr.npz")
        select_top_windows_with_snr(step3, step4)

        # 5) Compute SBP/DBP labels from the selected windows
        sbp_dbp_labels(step4, final_path)

        # TemporaryDirectory auto‐cleans all step1..4 files once we exit this block

    print(f"→ Finished processing '{base}'. Final saved to:\n   {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run end‐to‐end PPG→SBP/DBP pipeline on all raw .npz files in a folder."
    )
    parser.add_argument(
        "--raw_dir",
        required=True,
        help="Folder containing raw .npz files (each must have 'ppg','abp','fs=500')."
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Folder where final '_final.npz' outputs will be saved."
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Find all .npz files in raw_dir
    pattern = os.path.join(raw_dir, "*.npz")
    raw_files = sorted(glob.glob(pattern))

    if not raw_files:
        print(f"No .npz files found in '{raw_dir}'. Exiting.")
        return

    for raw_path in raw_files:
        try:
            run_pipeline_on_single_file(raw_path, out_dir)
        except Exception as e:
            base = os.path.basename(raw_path)
            print(f"⚠️  Error processing '{base}': {e}")


if __name__ == "__main__":
    main()

#chmod +x run_full_pipeline.py
#./run_full_pipeline.py --raw_dir /path/to/raw_data --out_dir /path/to/output_folder
