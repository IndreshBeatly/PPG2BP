#!/usr/bin/env python3
import argparse
import numpy as np
from typing import Tuple


def find_peaks_with_min_distance(
    sig: np.ndarray,
    min_prominence: float,
    fs: float,
    min_dist_s: float
) -> np.ndarray:
    """
    Identify local maxima in `sig` that exceed (global_min + min_prominence)
    and are separated by at least `min_dist_s` seconds.
    Returns their indices.
    """
    L = sig.shape[0]
    if L < 3:
        return np.array([], dtype=int)

    gmin = float(np.min(sig))
    threshold = gmin + min_prominence

    # Compare each interior sample to its neighbors
    left   = sig[:-2]
    center = sig[1:-1]
    right  = sig[2:]
    peaks_bool = (center > left) & (center > right) & (center >= threshold)
    candidate_idxs = np.nonzero(peaks_bool)[0] + 1  # +1 to offset into original array

    # Enforce minimum distance between kept peaks
    min_dist_samples = int(min_dist_s * fs)
    if min_dist_samples < 1:
        min_dist_samples = 1

    kept = []
    last_idx = -np.inf
    for idx in candidate_idxs:
        if idx - last_idx >= min_dist_samples:
            kept.append(idx)
            last_idx = idx

    return np.array(kept, dtype=int)


def find_troughs_between_peaks(sig: np.ndarray, peak_idxs: np.ndarray) -> np.ndarray:
    """
    Given sorted peak indices, find the local minima (troughs) between each pair of consecutive peaks.
    Returns trough indices.
    """
    troughs = []
    for i in range(len(peak_idxs) - 1):
        start_idx = peak_idxs[i]
        end_idx = peak_idxs[i + 1]
        if end_idx - start_idx <= 1:
            continue
        segment = sig[start_idx:end_idx + 1]
        trough_offset = np.argmin(segment)
        trough_idx = start_idx + trough_offset
        troughs.append(trough_idx)
    return np.array(troughs, dtype=int)


def compute_sbp_dbp(
    abp_win: np.ndarray,
    fs: float
) -> Tuple[float, float]:
    """
    Compute SBP and DBP for a single 10-second ABP strip:
      - SBP = global maximum of abp_win
      - DBP = median of local minima between consecutive systolic peaks
              (falls back to abp_win.min() if fewer than two peaks detected)
    """
    # SBP by simple global max
    sbp = float(np.max(abp_win))

    # DBP via local minima between beats
    # 1) detect systolic peaks with a modest prominence
    abp_range = float(np.nanmax(abp_win) - np.nanmin(abp_win))
    min_prominence_abp = 0.05 * abp_range
    min_dist_s = 0.3  # enforce at least 0.3 s between true peaks

    peak_idxs = find_peaks_with_min_distance(abp_win, min_prominence_abp, fs, min_dist_s)
    if peak_idxs.size < 2:
        # fallback to global min if not enough peaks found
        dbp = float(np.min(abp_win))
        return sbp, dbp

    # 2) find troughs between detected peaks
    trough_idxs = find_troughs_between_peaks(abp_win, peak_idxs)
    if trough_idxs.size == 0:
        dbp = float(np.min(abp_win))
    else:
        trough_vals = abp_win[trough_idxs]
        dbp = float(np.median(trough_vals))

    return sbp, dbp


def sbp_dbp_labels(input_npz: str, output_npz: str) -> None:
    """
    Load a .npz containing:
      • ppg_windows  (shape = [N, window_samples])
      • abp_windows  (shape = [N, window_samples])
      • fs           (scalar)
    For each i in [0..N-1]:
      sbp_values[i] = max(abp_windows[i, :])
      dbp_values[i] = median of local minima between peaks in abp_windows[i, :]
    Save out a new .npz with:
      • ppg_windows  (unchanged)
      • sbp_values   (shape = [N,])
      • dbp_values   (shape = [N,])
      • fs           (scalar)
    """
    data = np.load(input_npz)
    ppg_windows = data['ppg_windows']
    abp_windows = data['abp_windows']
    fs = float(data['fs'])
    data.close()

    N, window_samples = ppg_windows.shape
    sbp_values = np.zeros((N,), dtype=np.float32)
    dbp_values = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        abp_strip = abp_windows[i, :]
        sbp_i, dbp_i = compute_sbp_dbp(abp_strip, fs)
        sbp_values[i] = sbp_i
        dbp_values[i] = dbp_i

    np.savez_compressed(
        output_npz,
        ppg_windows=ppg_windows,
        sbp_values=sbp_values,
        dbp_values=dbp_values,
        fs=fs
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compute SBP (global max) and DBP (median of local minima) for each ABP strip."
    )
    parser.add_argument(
        "input_npz",
        help="Path to input .npz (must contain 'ppg_windows', 'abp_windows', and 'fs')."
    )
    parser.add_argument(
        "output_npz",
        help="Path where output .npz (with 'ppg_windows', 'sbp_values', 'dbp_values', 'fs') will be saved."
    )
    args = parser.parse_args()
    sbp_dbp_labels(args.input_npz, args.output_npz)
    print(f"Saved PPG windows with SBP/DBP labels to '{args.output_npz}'.")


if __name__ == "__main__":
    main()
