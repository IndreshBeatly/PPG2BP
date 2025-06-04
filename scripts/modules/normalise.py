#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_ppg(ppg: np.ndarray, fs: float,
                 lowcut: float = 0.5, highcut: float = 10.0,
                 order: int = 4) -> np.ndarray:
    """
    Apply a zero‐phase Butterworth bandpass filter to a 1D PPG signal.

    Args:
      ppg     : 1D numpy array of raw PPG samples (length = window_samples).
      fs      : Sampling frequency (e.g. 50.0).
      lowcut  : Lower cutoff frequency (Hz), default = 0.5.
      highcut : Upper cutoff frequency (Hz), default = 10.0.
      order   : Filter order, default = 4.

    Returns:
      1D numpy array of same length, bandpass‐filtered.
    """
    nyq = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return filtfilt(b, a, ppg)


def bandpass_and_normalize_ppg_windows(ppg_windows: np.ndarray, fs: float) -> np.ndarray:
    """
    For each PPG window (10 s @ 50 Hz), apply:
      1) 0.5–10 Hz Butterworth band‐pass filter (4th order, zero‐phase)
      2) Subtract that window's median
      3) Divide by that window's standard deviation

    Args:
      ppg_windows : 2D numpy array (N_windows × window_samples) of raw PPG.
      fs          : Sampling frequency for those windows (e.g. 50.0).

    Returns:
      A 2D numpy array of the same shape, filtered + normalized.
    """
    N, L = ppg_windows.shape
    out = np.zeros((N, L), dtype=np.float32)

    for i in range(N):
        raw_win = ppg_windows[i, :]
        # 1) Band‐pass filter
        bp = bandpass_ppg(raw_win, fs)
        # 2) Subtract median (center)
        med = np.median(bp)
        centered = bp - med
        # 3) Divide by std (z‐score)
        std = np.std(centered) + 1e-6
        out[i, :] = centered / std

    return out


def filter_ppg_file(input_npz: str, output_npz: str) -> None:
    """
    Load a .npz with keys:
      • 'ppg_windows'   (shape = N × window_samples)
      • 'sbp_values'
      • 'dbp_values'
      • 'fs'
    Apply band‐pass (0.5–10 Hz) + per‐window median‐centering and z‐score
    normalization, then save a new .npz containing:
      • ppg_windows   (filtered + normalized)
      • sbp_values
      • dbp_values
      • fs
    """
    data = np.load(input_npz)
    ppg_windows = data["ppg_windows"]  # shape = (N, 500)
    sbp_values  = data["sbp_values"]
    dbp_values  = data["dbp_values"]
    fs = float(data["fs"])
    data.close()

    print(f"[INFO] Bandpass‐filtering + normalizing {ppg_windows.shape[0]} PPG windows.")
    ppg_filtered_norm = bandpass_and_normalize_ppg_windows(ppg_windows, fs)

    np.savez_compressed(
        output_npz,
        ppg_windows=ppg_filtered_norm,
        sbp_values=sbp_values,
        dbp_values=dbp_values,
        fs=fs
    )
    print(f"[DONE] Saved filtered+normalized file to '{output_npz}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bandpass (0.5–10 Hz) and then normalize PPG windows from a .npz file."
    )
    parser.add_argument("input_npz", help="Path to input .npz (must contain 'ppg_windows').")
    parser.add_argument("output_npz", help="Path to save the filtered+normalized .npz file.")
    args = parser.parse_args()

    filter_ppg_file(args.input_npz, args.output_npz)
