#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_ppg(ppg: np.ndarray, fs: float,
                 lowcut: float = 0.5, highcut: float = 5.0,
                 order: int = 4) -> np.ndarray:
    """
    Apply a zero‐phase Butterworth bandpass filter to a 1D PPG signal.

    Args:
      ppg     : 1D numpy array of raw PPG samples (length = window_samples).
      fs      : Sampling frequency (e.g. 50.0).
      lowcut  : Lower cutoff frequency (Hz), default = 0.5.
      highcut : Upper cutoff frequency (Hz), default = 5.0.
      order   : Filter order, default = 4.

    Returns:
      1D numpy array of same length, bandpass‐filtered.
    """
    nyq = 0.5 * fs
    low  = lowcut  / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    # filtfilt does zero‐phase filtering
    return filtfilt(b, a, ppg)


def normalize_ppg(ppg_windows: np.ndarray, fs: float, method: str = "zscore") -> np.ndarray:
    """
    Bandpass each PPG window (0.5–5 Hz) and then normalize.

    Args:
      ppg_windows : 2D numpy array (N_windows × window_samples) of raw PPG.
      fs          : Sampling frequency for those windows (e.g. 50.0).
      method      : "zscore" (subtract median, divide by std) or "center" (subtract median only).

    Returns:
      A 2D numpy array of the same shape, filtered + normalized.
    """
    N, L = ppg_windows.shape
    filtered = np.zeros_like(ppg_windows, dtype=np.float32)

    # 1) Bandpass‐filter each window
    for i in range(N):
        filtered[i, :] = bandpass_ppg(ppg_windows[i, :], fs)

    # 2) Subtract median from each window
    median = np.median(filtered, axis=1, keepdims=True)
    centered = filtered - median

    if method == "center":
        return centered
    elif method == "zscore":
        std = np.std(centered, axis=1, keepdims=True) + 1e-6
        return centered / std
    else:
        raise ValueError(f"Unsupported normalization method: {method}")


def normalize_ppg_file(input_npz: str, output_npz: str, method: str = "zscore") -> None:
    """
    Load a .npz with 'ppg_windows', 'sbp_values', 'dbp_values', 'fs',
    bandpass‐filter + normalize each PPG window, then save a new .npz.

    The final file will contain:
      • ppg_windows  (filtered + normalized)
      • sbp_values
      • dbp_values
      • fs
    """
    data = np.load(input_npz)
    ppg_windows = data["ppg_windows"]  # shape = (N, window_samples)
    sbp_values = data["sbp_values"]
    dbp_values = data["dbp_values"]
    fs = float(data["fs"])
    data.close()

    print(f"[INFO] Bandpass + normalizing {ppg_windows.shape[0]} PPG windows using '{method}' method.")
    ppg_windows_norm = normalize_ppg(ppg_windows, fs, method=method)

    np.savez_compressed(
        output_npz,
        ppg_windows=ppg_windows_norm,
        sbp_values=sbp_values,
        dbp_values=dbp_values,
        fs=fs
    )
    print(f"[DONE] Saved filtered+normalized file to '{output_npz}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Bandpass (0.5–5 Hz) and normalize ppg_windows from a .npz file."
    )
    parser.add_argument("input_npz", help="Path to input .npz (must contain 'ppg_windows').")
    parser.add_argument("output_npz", help="Path to save the filtered+normalized .npz file.")
    parser.add_argument(
        "--method", choices=["zscore", "center"], default="zscore",
        help="Normalization method (default: zscore)."
    )
    args = parser.parse_args()

    normalize_ppg_file(args.input_npz, args.output_npz, method=args.method)


if __name__ == "__main__":
    main()
