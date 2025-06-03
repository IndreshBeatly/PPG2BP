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

    # 1) find all local maxima above threshold
    left   = sig[:-2]
    center = sig[1:-1]
    right  = sig[2:]
    peaks_bool = (center > left) & (center > right) & (center >= threshold)
    candidate_idxs = np.nonzero(peaks_bool)[0] + 1  # +1 because center = sig[1:-1]

    # 2) enforce a minimum distance (in samples) between kept peaks
    min_dist_samples = int(min_dist_s * fs)
    if min_dist_samples < 1:
        min_dist_samples = 1

    kept = []
    last_idx = -np.inf
    for idx in candidate_idxs:
        if idx - last_idx >= min_dist_samples:
            kept.append(idx)
            last_idx = idx
        # else: skip this idx because it’s too close to the previous kept peak

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
        # if the window is extremely flat, we can consider SNR undefined / too small
        return None

    peak_heights = sig[peak_idxs]
    median_peak = float(np.median(peak_heights))
    signal_amp = median_peak - median_val
    if signal_amp <= 0:
        return None

    return signal_amp / (mad + eps)


def select_top_windows_with_snr(input_path: str, output_path: str) -> None:
    """
    1) Load 'ppg', 'abp', and 'fs' from input_path (.npz).
    2) Split into non‐overlapping 10 s windows: window_samples = int(fs * 10).
    3) For each window:
         a) Detect raw peaks in PPG/ABP with a short refractory period.
         b) If either channel has fewer than `min_peaks_per_window` peaks, skip.
         c) Compute SNR_ppg = compute_snr(ppg_win, ppg_peaks)
            and    SNR_abp = compute_snr(abp_win, abp_peaks).
            If either is None or < `min_snr`, skip.
         d) Score = SNR_ppg + SNR_abp. Push (score, start_idx) into a min‐heap of size ≤ 100.
    4) At the end, extract up to 100 highest‐score windows, sort by descending score,
       collect their slices, and save:
         • ppg_windows: shape (N_sel, window_samples)
         • abp_windows: shape (N_sel, window_samples)
         • starts:      shape (N_sel,)
         • snr_scores:  shape (N_sel,)  # combined (ppg+abp) SNR
         • fs:          scalar
    """
    data = np.load(input_path)
    ppg = data["ppg"]       # 1D array of length L
    abp = data["abp"]       # 1D array of length L
    fs  = float(data["fs"]) # e.g. 50.0 (or 500.0, but ideally downsampled)

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
        raise RuntimeError(
            f"Signal too short for a 10 s window: need ≥ {window_samples} samples, got {total_samples}."
        )

    # --------- PARAMETERS YOU CAN ADJUST ----------
    # Rough “prominence” threshold as a percentage of each signal’s global range:
    min_prominence_ppg = 0.05 * (np.nanmax(ppg) - np.nanmin(ppg))
    min_prominence_abp = 0.05 * (np.nanmax(abp) - np.nanmin(abp))

    # Minimum number of raw peaks per 10 s window (e.g. 3 → ~18 bpm floor)
    min_peaks_per_window = 3

    # Minimum SNR in each channel to count the window
    min_snr = 2.0

    # Refractory period for raw peak detector (seconds)
    min_dist_s = 0.2  # signals closer than 0.2 s will be collapsed

    # -----------------------------------------------

    # Min‐heap for (combined_snr, start_idx), size ≤ 100
    heap: list[tuple[float, int]] = []
    max_kept = 300

    for widx in range(num_windows):
        start = widx * window_samples
        end   = start + window_samples
        ppg_win = ppg[start:end]
        abp_win = abp[start:end]

        # 1) Detect “raw” peaks with refractory period = min_dist_s
        ppg_peaks = find_peaks_with_min_distance(ppg_win, min_prominence_ppg, fs, min_dist_s)
        abp_peaks = find_peaks_with_min_distance(abp_win, min_prominence_abp, fs, min_dist_s)

        # 2) Skip if too few raw peaks
        if ppg_peaks.size < min_peaks_per_window or abp_peaks.size < min_peaks_per_window:
            continue

        # 3) Compute SNR in each channel
        snr_ppg = compute_snr(ppg_win, ppg_peaks)
        snr_abp = compute_snr(abp_win, abp_peaks)
        if (snr_ppg is None) or (snr_abp is None):
            continue
        if snr_ppg < min_snr or snr_abp < min_snr:
            continue

        # 4) Combined score and heap logic
        combined_snr = snr_ppg + snr_abp
        max_snr = 50.0
        # skip if combinedsnr is higher than max_snr
        if combined_snr > max_snr:
            continue
        if len(heap) < max_kept:
            heapq.heappush(heap, (combined_snr, start))
        else:
            if combined_snr > heap[0][0]:
                heapq.heapreplace(heap, (combined_snr, start))

    if not heap:
        raise RuntimeError("No valid 10 s windows found after SNR filtering.")

    # Extract top windows, sorted by descending SNR
    top_list = sorted(heap, key=lambda x: -x[0])  # [(combined_snr, start), …]
    n_sel = len(top_list)

    # Pre‐allocate arrays for the selected windows
    ppg_sel = np.zeros((n_sel, window_samples), dtype=ppg.dtype)
    abp_sel = np.zeros((n_sel, window_samples), dtype=abp.dtype)
    starts  = np.zeros((n_sel,), dtype=np.int64)
    scores  = np.zeros((n_sel,), dtype=np.float32)

    for i, (snr_val, st) in enumerate(top_list):
        ppg_sel[i, :] = ppg[st : st + window_samples]
        abp_sel[i, :] = abp[st : st + window_samples]
        starts[i] = st
        scores[i] = snr_val

    # Clean up large arrays
    del ppg, abp, data, heap, top_list

    # Save results
    np.savez_compressed(
        output_path,
        ppg_windows=ppg_sel,
        abp_windows=abp_sel,
        starts=starts,
        snr_scores=scores,
        fs=fs
    )


def main():
    parser = argparse.ArgumentParser(
        description="Select top‐100 windows based on SNR in PPG & ABP signals."
    )
    parser.add_argument(
        "input_npz",
        help=(
            "Path to the input .npz (must contain 'ppg', 'abp', and 'fs').\n"
            "You should already have stripped head/tail, removed NaNs, and downsampled."
        )
    )
    parser.add_argument(
        "output_npz",
        help="Path where the top‐100 SNR‐filtered windows (.npz) will be saved."
    )
    args = parser.parse_args()
    select_top_windows_with_snr(args.input_npz, args.output_npz)
    print(f"Saved top windows (SNR‐based) to '{args.output_npz}'.")


if __name__ == "__main__":
    main()
