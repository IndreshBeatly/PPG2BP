#!/usr/bin/env python3
import argparse
import numpy as np

def sbp_dbp_minmax(
    input_npz: str,
    output_npz: str
) -> None:
    """
    Load an .npz with 'ppg_windows', 'abp_windows', and 'fs'.
    For each 10 s ABP strip:
      - SBP = np.max(abp_strip)
      - DBP = np.min(abp_strip)
    Save out a new .npz containing:
      • ppg_windows   : shape (N, window_samples)
      • sbp_values    : shape (N,)
      • dbp_values    : shape (N,)
      • fs            : scalar
    """
    data = np.load(input_npz)
    ppg_windows = data['ppg_windows']   # shape = (N, window_samples)
    abp_windows = data['abp_windows']   # shape = (N, window_samples)
    fs = float(data['fs'])
    data.close()

    N, window_samples = ppg_windows.shape

    # Preallocate label arrays:
    sbp_values = np.zeros((N,), dtype=np.float32)
    dbp_values = np.zeros((N,), dtype=np.float32)

    # Compute SBP/DBP by global max/min on each ABP window:
    for i in range(N):
        abp_strip = abp_windows[i, :]
        sbp_values[i] = float(np.max(abp_strip))
        dbp_values[i] = float(np.min(abp_strip))

    # Save results:
    np.savez_compressed(
        output_npz,
        ppg_windows=ppg_windows,
        sbp_values=sbp_values,
        dbp_values=dbp_values,
        fs=fs
    )

def main():
    parser = argparse.ArgumentParser(
        description="Generate SBP=global max and DBP=global min for each ABP strip."
    )
    parser.add_argument(
        "input_npz",
        help="Path to input .npz (must contain 'ppg_windows', 'abp_windows', and 'fs')."
    )
    parser.add_argument(
        "output_npz",
        help="Path where the output .npz (with 'ppg_windows', 'sbp_values', 'dbp_values', 'fs') will be saved."
    )
    args = parser.parse_args()
    sbp_dbp_minmax(args.input_npz, args.output_npz)
    print(f"Saved PPG windows with SBP/DBP (min/max) labels to '{args.output_npz}'.")

if __name__ == "__main__":
    main()
