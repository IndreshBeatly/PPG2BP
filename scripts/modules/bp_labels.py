#!/usr/bin/env python3
import argparse
import numpy as np
from typing import Tuple
import neurokit2 as nk


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
    troughs = []
    for i in range(len(peak_idxs) - 1):
        start_idx = peak_idxs[i]
        end_idx   = peak_idxs[i + 1]
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
    SBP = mean of NeuroKit2-detected PPG_Peaks (fallback to global max)
    DBP = median of NeuroKit2-detected PPG_Valleys (fallback to troughs, then global min)
    """
    # 1) NeuroKit peak/valley detection
    info = nk.ppg_findpeaks(abp_win, sampling_rate=fs, method="elgendi")
    peak_idxs   = info.get("PPG_Peaks",   np.array([], dtype=int))
    valley_idxs = info.get("PPG_Valleys", np.array([], dtype=int))

    # 2) SBP
    if peak_idxs.size > 0:
        sbp = float(np.mean(abp_win[peak_idxs]))
    else:
        sbp = float(np.max(abp_win))

    # 3) DBP
    if valley_idxs.size > 0:
        dbp = float(np.median(abp_win[valley_idxs]))
    else:
        trough_idxs = find_troughs_between_peaks(abp_win, peak_idxs)
        if trough_idxs.size > 0:
            dbp = float(np.median(abp_win[trough_idxs]))
        else:
            dbp = float(np.min(abp_win))

    return sbp, dbp

def sbp_dbp_labels(input_npz: str, output_npz: str) -> None:
    """
    Load a .npz containing:
      • ppg_windows  (shape = [N, window_samples])
      • abp_windows  (shape = [N, window_samples])
      • fs           (scalar)
    Compute SBP/DBP for each strip and save results.
    """
    data = np.load(input_npz)
    ppg_windows = data['ppg_windows']
    abp_windows = data['abp_windows']
    fs = float(data['fs'])
    data.close()

    N,window_samples = ppg_windows.shape
    sbp_values = np.zeros((N,), dtype=np.float32)
    dbp_values = np.zeros((N,), dtype=np.float32)

    for i in range(N):
        sbp_i, dbp_i = compute_sbp_dbp(abp_windows[i, :], fs)
        sbp_values[i] = sbp_i
        dbp_values[i] = dbp_i

    np.savez_compressed(
        output_npz,
        ppg_windows=ppg_windows,
        sbp_values=sbp_values,
        dbp_values=dbp_values,
        fs=fs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute SBP (mean of peaks) and DBP (median of valleys) for each ABP strip."
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
