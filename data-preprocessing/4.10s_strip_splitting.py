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
    Return indices of “true” peaks in `sig` such that:
      • Each peak is a local maxima ≥ (global_min + min_prominence).
      • No two kept peaks are closer than `min_dist_s` seconds.
    """
    L = sig.shape[0]
    if L < 3:
        return np.array([], dtype=int)

    gmin = float(np.min(sig))
    threshold = gmin + min_prominence

    # Step 1: identify all local maxima above threshold
    left   = sig[:-2]
    center = sig[1:-1]
    right  = sig[2:]
    peaks_bool = (center > left) & (center > right) & (center >= threshold)
    candidate_idxs = np.nonzero(peaks_bool)[0] + 1  # +1 because 'center' = sig[1:-1]

    # Step 2: enforce minimum distance (in samples) between kept peaks
    min_dist_samples = int(min_dist_s * fs)
    if min_dist_samples < 1:
        min_dist_samples = 1

    kept = []
    last_idx = -np.inf
    for idx in candidate_idxs:
        if idx - last_idx >= min_dist_samples:
            kept.append(idx)
            last_idx = idx
        # else: skip this idx because it's too close to the previous kept peak

    return np.array(kept, dtype=int)


def is_interval_consistent(
    peak_idxs: np.ndarray,
    fs: float,
    cv_thresh: float = 0.4  # (was 0.3 before; now more permissive)
) -> bool:
    """
    Return True if the inter-beat intervals in `peak_idxs` are reasonably consistent.
    Compute IBIs = diff(peak_idxs / fs) and then coefficient of variation.
    If CV < cv_thresh, we say “consistent”; otherwise “too irregular → discard.”
    """
    if peak_idxs.size < 3:
        # Fewer than 3 peaks → not enough data for a reliable IBI check
        return False

    times = peak_idxs.astype(float) / fs
    ibis = np.diff(times)
    mean_ibi = np.mean(ibis)
    std_ibi  = np.std(ibis)
    if mean_ibi <= 0:
        return False

    cv = std_ibi / mean_ibi
    return (cv < cv_thresh)


def count_matched_peaks(
    ppg_peaks: np.ndarray,
    abp_peaks: np.ndarray,
    max_lag_s: float,
    fs: float
) -> int:
    """
    Return the number of 1:1 matches between ppg_peaks and abp_peaks if
    |idx_abp - idx_ppg| ≤ max_lag_s * fs. Once an ABP peak is matched, it cannot match again.
    """
    if ppg_peaks.size == 0 or abp_peaks.size == 0:
        return 0

    max_lag = int(max_lag_s * fs)
    ppg_sorted = np.sort(ppg_peaks)
    abp_sorted = np.sort(abp_peaks)
    matched = 0
    j = 0
    N_abp = abp_sorted.shape[0]

    for i_ppg in ppg_sorted:
        # Advance j until abp_sorted[j] ≥ (i_ppg - max_lag)
        while j < N_abp and abp_sorted[j] < i_ppg - max_lag:
            j += 1
        if j >= N_abp:
            break

        # Now abp_sorted[j] is the first ABP peak ≥ (i_ppg - max_lag)
        if abs(int(abp_sorted[j]) - int(i_ppg)) <= max_lag:
            matched += 1
            j += 1  # consume this ABP peak so it won’t match again
        # If abp_sorted[j] > i_ppg + max_lag, no match for this PPG peak → move on

    return matched


def compute_composite_score(
    ppg_win: np.ndarray,
    abp_win: np.ndarray,
    ppg_peaks: np.ndarray,
    abp_peaks: np.ndarray,
    fs: float,
    min_peaks_per_window: int
) -> Optional[float]:
    """
    Compute a composite “quality” score for a 10 s window:
      1) Count matched beats between ppg_peaks and abp_peaks (±0.1 s).
         If fewer than min_peaks_per_window matched, return None.
      2) Check inter‐beat‐interval (IBI) consistency on PPG. If CV > 0.4, return None.
      3) Compute amplitude‐consistency penalty: (max_peak - min_peak)/median_peak.
      4) (Optional) cross-correlation check (commented out by default).
      5) Combine normalized matched beats, interval consistency, and amplitude consistency
         into a final score in [0, 1].
    """
    window_seconds = ppg_win.shape[0] / fs

    # 1) matched beats
    matched = count_matched_peaks(ppg_peaks, abp_peaks, max_lag_s=0.1, fs=fs)
    if matched < min_peaks_per_window:
        return None  # too few matched beats → discard

    # 2) IBI consistency (PPG)
    times_ppg = ppg_peaks.astype(float) / fs
    ibis_ppg = np.diff(times_ppg)
    if ibis_ppg.size < 2:
        return None
    mean_ibi = np.mean(ibis_ppg)
    std_ibi  = np.std(ibis_ppg)
    if mean_ibi <= 0:
        return None
    cv_ppg = std_ibi / mean_ibi
    if cv_ppg > 0.4:
        return None  # too irregular → discard
    score_interval = max(0.0, 1.0 - (cv_ppg / 0.4))

    # 3) amplitude consistency
    heights_ppg = ppg_win[ppg_peaks]
    heights_abp = abp_win[abp_peaks]
    if heights_ppg.size < 1 or heights_abp.size < 1:
        return None
    amp_ratio_ppg = (
        (float(np.max(heights_ppg)) - float(np.min(heights_ppg)))
        / float(np.median(heights_ppg))
    )
    amp_ratio_abp = (
        (float(np.max(heights_abp)) - float(np.min(heights_abp)))
        / float(np.median(heights_abp))
    )
    score_amp = 1.0 / (1.0 + amp_ratio_ppg + amp_ratio_abp)

    # 4) (optional) cross‐correlation
    # corr = np.corrcoef(ppg_win, abp_win)[0, 1]
    # score_corr = max(0.0, (corr - 0.2) / 0.8)

    # 5) normalize matched beats:
    #    assume a physiological upper bound of ~3 beats/sec (180 bpm) per channel:
    max_possible_beats = window_seconds * 3.0
    norm_matched = matched / max_possible_beats
    if norm_matched > 1.0:
        norm_matched = 1.0

    # Combine weights: 60% matched, 20% interval consistency, 20% amplitude consistency
    final_score = 0.6 * norm_matched + 0.2 * score_interval + 0.2 * score_amp
    # If you want cross‐corr, add something like +0.1 * score_corr and reduce other weights.
    return final_score


def select_top_windows_peak(input_path: str, output_path: str) -> None:
    """
    1) Load 'ppg', 'abp', and 'fs' from input_path (.npz).
    2) Split into non‐overlapping 10 s windows: window_samples = int(fs * 10).
    3) For each window:
         a) Detect raw peaks in PPG/ABP with refractory period (min_dist_s = 0.25 s).
         b) If either channel has fewer than 4 raw peaks, skip.
         c) Compute composite score (matched beats, IBI consistency, amplitude consistency).
            If score is None, skip.
         d) Maintain a min‐heap (size ≤ 100) of (score, start_idx).
    4) Extract up to 100 highest‐score windows, sort by descending score,
       collect slices, and save:
         • ppg_windows: shape (N_sel, window_samples)
         • abp_windows: shape (N_sel, window_samples)
         • starts:     array (N_sel,)
         • scores:     array (N_sel,)
         • fs:         scalar
    """
    data = np.load(input_path)
    ppg = data["ppg"]       # 1D array, length L
    abp = data["abp"]       # 1D array, length L
    fs = float(data["fs"])  # e.g. 50.0 (ideally downsampled)

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

    # Looser heuristic thresholds (tune further if needed)
    #  • min_prominence based on entire-signal range
    #  • now 5% of range instead of 10%
    min_prominence_ppg = 0.05 * (np.nanmax(ppg) - np.nanmin(ppg))
    min_prominence_abp = 0.05 * (np.nanmax(abp) - np.nanmin(abp))
    #  • require only 4 peaks per 10 s (≈ 24 bpm) instead of 6
    min_peaks_per_window = 4

    # Min-heap of (score, start_idx), size ≤ 100
    heap: list[tuple[float, int]] = []
    max_kept = 100

    for widx in range(num_windows):
        start = widx * window_samples
        end = start + window_samples
        ppg_win = ppg[start:end]
        abp_win = abp[start:end]

        # 1) Detect raw peaks with refractory period = 0.25 s (instead of 0.3 s)
        ppg_peaks = find_peaks_with_min_distance(
            ppg_win, min_prominence_ppg, fs, min_dist_s=0.25
        )
        abp_peaks = find_peaks_with_min_distance(
            abp_win, min_prominence_abp, fs, min_dist_s=0.25
        )

        # 2) Skip if too few raw peaks (now 4 instead of 6)
        if ppg_peaks.size < min_peaks_per_window or abp_peaks.size < min_peaks_per_window:
            continue

        # 3) Compute composite score; if None, skip
        score = compute_composite_score(
            ppg_win, abp_win, ppg_peaks, abp_peaks, fs, min_peaks_per_window
        )
        if score is None:
            continue

        # 4) Maintain min‐heap of size ≤ 100
        if len(heap) < max_kept:
            heapq.heappush(heap, (score, start))
        else:
            if score > heap[0][0]:
                heapq.heapreplace(heap, (score, start))

    if not heap:
        raise RuntimeError("No valid 10 s windows found after applying all checks.")

    # Extract top windows, sort by descending score
    top_list = sorted(heap, key=lambda x: -x[0])  # [(score, start), …]
    n_sel = len(top_list)

    ppg_sel = np.zeros((n_sel, window_samples), dtype=ppg.dtype)
    abp_sel = np.zeros((n_sel, window_samples), dtype=abp.dtype)
    starts = np.zeros((n_sel,), dtype=np.int64)
    scores = np.zeros((n_sel,), dtype=np.float32)

    for i, (sc, st) in enumerate(top_list):
        ppg_sel[i, :] = ppg[st : st + window_samples]
        abp_sel[i, :] = abp[st : st + window_samples]
        starts[i] = st
        scores[i] = sc

    # Release large arrays ASAP
    del ppg, abp, data, heap, top_list

    # Save results
    np.savez_compressed(
        output_path,
        ppg_windows=ppg_sel,
        abp_windows=abp_sel,
        starts=starts,
        scores=scores,
        fs=fs
    )


def main():
    parser = argparse.ArgumentParser(
        description="Select top‐100 “good” 10 s windows from PPG/ABP using a more‐permissive peak-based method."
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
        help="Path where the top‐100 windows (.npz) will be saved."
    )
    args = parser.parse_args()
    select_top_windows_peak(args.input_npz, args.output_npz)
    print(f"Saved top windows (more‐permissive) to '{args.output_npz}'.")


if __name__ == "__main__":
    main()
