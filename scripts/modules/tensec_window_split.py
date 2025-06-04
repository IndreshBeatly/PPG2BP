#!/usr/bin/env python3
import argparse
import heapq
import numpy as np
from typing import Optional

def find_peaks_with_min_distance(
    sig: np.ndarray,
    min_prominence: float,
    fs: float,
    min_dist_s: float
) -> np.ndarray:
    """
    Return indices of “raw” peaks in `sig` such that:
      • Each candidate peak is a local maxima ≥ (global_min + min_prominence).
      • No two kept peaks are closer than `min_dist_s` seconds.
    """
    L = sig.shape[0]
    if L < 3:
        return np.array([], dtype=int)

    gmin = float(np.min(sig))
    threshold = gmin + min_prominence

    left   = sig[:-2]
    center = sig[1:-1]
    right  = sig[2:]
    peaks_bool = (center > left) & (center > right) & (center >= threshold)
    candidate_idxs = np.nonzero(peaks_bool)[0] + 1  # +1 because center = sig[1:-1]

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


def compute_snr(sig: np.ndarray, peak_idxs: np.ndarray, eps: float = 1e-6) -> Optional[float]:
    """
    Compute a rough SNR for `sig` on a single window:
      • Baseline = median(sig)
      • Noise estimate  = MAD = median(|sig - median(sig)|)
      • “Signal amplitude” = median(sig[peak_idxs]) - median(sig)
      • SNR = (median_peak_height - baseline) / (MAD + eps)

    Returns None if there are no peaks or if MAD is zero.
    """
    if peak_idxs.size == 0:
        return None

    median_val = float(np.median(sig))
    mad = float(np.median(np.abs(sig - median_val)))
    if mad < eps:
        return None

    peak_heights = sig[peak_idxs]
    median_peak = float(np.median(peak_heights))
    signal_amp = median_peak - median_val
    if signal_amp <= 0:
        return None

    return signal_amp / (mad + eps)


def select_top_windows_with_snr(
    input_path: str,
    output_path: str,
    min_windows: int = 50
) -> bool:
    """
    1) Load 'ppg', 'abp', and 'fs' from input_path (.npz).
    2) Split into non‐overlapping 10 s windows: window_samples = int(fs * 10).
    3) For each window, apply peak‐count and SNR filtering. Count how many
       windows survive all filters; if that count < min_windows, print
       "has_enough_windows: False" and return False immediately.
    4) Otherwise, take up to 300 of those surviving windows with highest
       combined SNR, save them to output_path, print "has_enough_windows: True",
       and return True.
    """
    data = np.load(input_path)
    ppg = data["ppg"]       # 1D array of length L
    abp = data["abp"]       # 1D array of length L
    fs  = float(data["fs"]) # e.g. 50.0 (or 500.0, but ideally already downsampled)
    data.close()

    if ppg.shape != abp.shape:
        raise ValueError("PPG and ABP must have the same shape.")
    total_samples = ppg.shape[0]

    # 10-second window in samples
    window_seconds = 10
    window_samples = int(fs * window_seconds)
    if window_samples <= 0:
        raise ValueError(f"Invalid fs={fs} ⇒ window_samples={window_samples}")

    num_windows = total_samples // window_samples
    if num_windows == 0:
        # No windows can be formed at all
        print("has_enough_windows: False")
        return False

    # --------- PARAMETERS YOU CAN ADJUST ----------
    min_prominence_ppg = 0.05 * (np.nanmax(ppg) - np.nanmin(ppg))
    min_prominence_abp = 0.05 * (np.nanmax(abp) - np.nanmin(abp))
    min_peaks_per_window = 3
    min_snr = 2.0
    min_dist_s = 0.2  # seconds
    max_combined_snr = 50.0
    max_kept = 300
    # -----------------------------------------------

    # First pass: count how many windows survive the peak + SNR criteria
    valid_count = 0
    for widx in range(num_windows):
        start = widx * window_samples
        end   = start + window_samples
        ppg_win = ppg[start:end]
        abp_win = abp[start:end]

        # Detect raw peaks
        ppg_peaks = find_peaks_with_min_distance(ppg_win, min_prominence_ppg, fs, min_dist_s)
        abp_peaks = find_peaks_with_min_distance(abp_win, min_prominence_abp, fs, min_dist_s)
        if ppg_peaks.size < min_peaks_per_window or abp_peaks.size < min_peaks_per_window:
            continue

        # Compute SNR
        snr_ppg = compute_snr(ppg_win, ppg_peaks)
        snr_abp = compute_snr(abp_win, abp_peaks)
        if (snr_ppg is None) or (snr_abp is None):
            continue
        if snr_ppg < min_snr or snr_abp < min_snr:
            continue

        combined_snr = snr_ppg + snr_abp
        if combined_snr > max_combined_snr:
            continue

        valid_count += 1

    # If too few valid windows, bail out
    if valid_count < min_windows:
        print("has_enough_windows: False")
        return False
    else:
        print("has_enough_windows: True")

    # Second pass: build a heap of up to max_kept windows by combined SNR
    heap: list[tuple[float, int]] = []
    for widx in range(num_windows):
        start = widx * window_samples
        end   = start + window_samples
        ppg_win = ppg[start:end]
        abp_win = abp[start:end]

        # Detect raw peaks
        ppg_peaks = find_peaks_with_min_distance(ppg_win, min_prominence_ppg, fs, min_dist_s)
        abp_peaks = find_peaks_with_min_distance(abp_win, min_prominence_abp, fs, min_dist_s)
        if ppg_peaks.size < min_peaks_per_window or abp_peaks.size < min_peaks_per_window:
            continue

        # Compute SNR
        snr_ppg = compute_snr(ppg_win, ppg_peaks)
        snr_abp = compute_snr(abp_win, abp_peaks)
        if (snr_ppg is None) or (snr_abp is None):
            continue
        if snr_ppg < min_snr or snr_abp < min_snr:
            continue

        combined_snr = snr_ppg + snr_abp
        if combined_snr > max_combined_snr:
            continue

        # Push into min‐heap of size ≤ max_kept
        if len(heap) < max_kept:
            heapq.heappush(heap, (combined_snr, start))
        else:
            if combined_snr > heap[0][0]:
                heapq.heapreplace(heap, (combined_snr, start))

    if not heap:
        # (Should never happen if valid_count >= min_windows, but just in case)
        raise RuntimeError("No valid 10 s windows found after SNR filtering.")

    # Extract top windows sorted by descending SNR
    top_list = sorted(heap, key=lambda x: -x[0])  # [(combined_snr, start), …]
    n_sel = len(top_list)

    ppg_sel = np.zeros((n_sel, window_samples), dtype=ppg.dtype)
    abp_sel = np.zeros((n_sel, window_samples), dtype=abp.dtype)
    starts  = np.zeros((n_sel,), dtype=np.int64)
    scores  = np.zeros((n_sel,), dtype=np.float32)

    for i, (snr_val, st) in enumerate(top_list):
        ppg_sel[i, :] = ppg[st : st + window_samples]
        abp_sel[i, :] = abp[st : st + window_samples]
        starts[i] = st
        scores[i] = snr_val

    # Save results
    np.savez_compressed(
        output_path,
        ppg_windows=ppg_sel,
        abp_windows=abp_sel,
        starts=starts,
        snr_scores=scores,
        fs=fs
    )

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select top windows based on SNR in PPG/ABP signals, only if #valid_windows ≥ min_windows."
    )
    parser.add_argument(
        "input_npz",
        help=(
            "Path to the input .npz (must contain 'ppg', 'abp', and 'fs').\n"
            "You should have already stripped head/tail, removed NaNs, and downsampled."
        )
    )
    parser.add_argument(
        "output_npz",
        help="Path where the SNR‐filtered windows (.npz) will be saved (if enough valid windows exist)."
    )
    parser.add_argument(
        "--min_windows",
        type=int,
        default=50,
        help="Minimum number of *valid* 10 s windows required; if fewer, exits early and returns False."
    )
    args = parser.parse_args()
    select_top_windows_with_snr(args.input_npz, args.output_npz, args.min_windows)
