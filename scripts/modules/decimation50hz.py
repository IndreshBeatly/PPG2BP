#!/usr/bin/env python3
import argparse
import numpy as np
from scipy.signal import butter, filtfilt

def lowpass_filter(x: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    """
    Zero‐phase low‐pass filter: attenuate all frequencies above 'cutoff_hz'.
    """
    nyq = fs / 2
    b, a = butter(order, cutoff_hz / nyq, btype="lowpass")
    return filtfilt(b, a, x)

def downsample_to_50hz(input_path: str, output_path: str) -> None:
    """
    1) Load 'ppg', 'abp', 'fs' from input_path (.npz),
    2) Low‐pass both at cutoff ~15 Hz to remove >25 Hz content,
    3) Decimate by taking every 10th sample (500 -> 50 Hz),
    4) Save as new .npz with fs=50.0.
    """
    data = np.load(input_path)
    ppg = data["ppg"]
    abp = data["abp"]
    fs  = float(data["fs"])

    if ppg.shape != abp.shape:
        raise ValueError("'ppg' and 'abp' must have the same shape.")
    if int(fs) != 500:
        raise ValueError(f"Expected original fs=500 Hz, but got fs={fs}.")

    # 1) Anti‐alias via low‐pass (zero‐phase) at 15 Hz
    ppg_filt = lowpass_filter(ppg, cutoff_hz=15.0, fs=fs, order=4)
    abp_filt = lowpass_filter(abp, cutoff_hz=15.0, fs=fs, order=4)

    # 2) Downsample by integer factor 10
    ppg_ds = ppg_filt[::10]
    abp_ds = abp_filt[::10]

    # 3) Save new 50 Hz file
    np.savez_compressed(
        output_path,
        ppg = ppg_ds,
        abp = abp_ds,
        fs  = float(50.0)
    )
    print(f"Saved downsampled signals to '{output_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample PPG/ABP from 500 Hz to 50 Hz (with low‐pass)."
    )
    parser.add_argument(
        "input_npz",
        help="Path to the cleaned .npz (must contain 'ppg','abp','fs=500')."
    )
    parser.add_argument(
        "output_npz",
        help="Where to save the downsampled .npz (ppg,abp at 50 Hz + 'fs')."
    )
    args = parser.parse_args()
    downsample_to_50hz(args.input_npz, args.output_npz)
