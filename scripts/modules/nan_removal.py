# now with the new .npz file , find a optimal way to remove only the nan indices , even if it is present either in one of the signal , abp or ppg , correspondingly 
# save the new file as another .npz file 


#!/usr/bin/env python3
import argparse
import numpy as np

def remove_nans(input_path: str, output_path: str) -> None:
    """
    Load 'ppg', 'abp', and 'fs' from input_path (.npz),
    remove all indices where either 'ppg' or 'abp' is NaN,
    and save the clean arrays (plus 'fs') into output_path (.npz).
    """
    data = np.load(input_path)
    ppg = data["ppg"]       # shape: (M,)
    abp = data["abp"]       # shape: (M,)
    fs = float(data["fs"])  # sampling frequency (unchanged)

    if ppg.shape != abp.shape:
        raise ValueError("'ppg' and 'abp' must have the same shape.")

    # Build a boolean mask of valid samples
    # True where neither is NaN
    valid_mask = (~np.isnan(ppg)) & (~np.isnan(abp))

    # Apply mask to both arrays; this creates new (smaller) arrays
    ppg_clean = ppg[valid_mask]
    abp_clean = abp[valid_mask]

    # Release memory of originals ASAP
    del ppg, abp, valid_mask, data

    # Save the cleaned signals with fs
    np.savez_compressed(
        output_path,
        ppg=ppg_clean,
        abp=abp_clean,
        fs=fs
    )
if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        description="Remove any NaN samples from PPG/ABP signals in a .npz file."
    )
    parser.add_argument(
        "input_npz",
        help="Path to the stripped .npz (from script #1) containing 'ppg', 'abp', and 'fs'."
    )
    parser.add_argument(
        "output_npz",
        help="Path where the NaN‐removed .npz will be saved."
    )
    args = parser.parse_args()
    remove_nans(args.input_npz, args.output_npz)
    print(f"Saved NaN‐free signals to '{args.output_npz}'.")




#chmod +x remove_nans.py
#./remove_nans.py stripped_file.npz cleaned_file.npz
