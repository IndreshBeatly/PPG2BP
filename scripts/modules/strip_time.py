# strip the 1st 30mins and last 10 mins of the signals file for both ppg and abp signals 
# save the output as the new .npz file 


#!/usr/bin/env python3
import argparse
import numpy as np

def strip_times(input_path: str, output_path: str) -> None:
    """
    Load 'ppg', 'abp', and 'fs' from input_path (.npz),
    strip the first 30 minutes and last 10 minutes of each signal,
    and save the result (plus 'fs') into output_path (.npz).
    """
    # Load data
    data = np.load(input_path)
    ppg = data["ppg"]       # shape: (N,)
    abp = data["abp"]       # shape: (N,)
    fs = float(data["fs"])  # sampling frequency, e.g. 500.0

    # Compute how many samples to strip
    strip_start = int(fs * 60 * 30)  # first 30 minutes
    strip_end = int(fs * 60 * 10)    # last 10 minutes

    if ppg.ndim != 1 or abp.ndim != 1:
        raise ValueError("Expected 1D 'ppg' and 'abp' arrays.")

    total_samples = ppg.shape[0]
    if total_samples <= strip_start + strip_end:
        raise ValueError(
            f"Signal length ({total_samples} samples) is too short to strip "
            f"{strip_start} + {strip_end} = {strip_start + strip_end} samples."
        )

    # Slice out unwanted segments (these are views, not copies)
    ppg_stripped = ppg[strip_start : total_samples - strip_end]
    abp_stripped = abp[strip_start : total_samples - strip_end]

    # Release memory of the original arrays as soon as possible
    del ppg, abp, data

    # Save the stripped signals and fs
    np.savez_compressed(
        output_path,
        ppg=ppg_stripped,
        abp=abp_stripped,
        fs=fs
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Strip first 30 min and last 10 min from PPG/ABP signals in a .npz file."
    )
    parser.add_argument(
        "input_npz",
        help="Path to the original .npz containing 'ppg', 'abp', and 'fs'."
    )
    parser.add_argument(
        "output_npz",
        help="Path where the stripped .npz will be saved."
    )
    args = parser.parse_args()
    strip_times(args.input_npz, args.output_npz)
    print(f"Saved stripped signals to '{args.output_npz}'.")


#chmod +x strip_times.py
#./strip_times.py input_file.npz stripped_file.npz
