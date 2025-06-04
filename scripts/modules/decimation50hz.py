# downsample the result from 500hz to 50hz and save as a new .npz file 
#!/usr/bin/env python3
import argparse
import numpy as np

def downsample_to_50hz(input_path: str, output_path: str) -> None:
    """
    Load 'ppg', 'abp', and 'fs' from input_path (.npz),
    downsample both signals from 500 Hz to 50 Hz by taking every 10th sample,
    and save the downsampled arrays (plus 'fs'=50) into output_path (.npz).
    """
    data = np.load(input_path)
    ppg = data["ppg"]       # shape: (L,)
    abp = data["abp"]       # shape: (L,)
    fs = float(data["fs"])  # sampling frequency, expected 500.0

    if ppg.shape != abp.shape:
        raise ValueError("'ppg' and 'abp' must have the same shape.")

    if int(fs) != 500:
        raise ValueError(
            f"Expected original fs=500 Hz, but got fs={fs}."
        )

    # Compute downsampling factor
    target_fs = 50
    factor = int(fs // target_fs)
    if fs / factor != target_fs:
        raise ValueError(
            f"Downsampling factor must be an integer. Got fs={fs}, "
            f"target_fs={target_fs}, fs/factor={fs/factor}."
        )

    # Use slicing to take every 'factor'-th sample (view, memory-efficient)
    ppg_ds = ppg[::factor]
    abp_ds = abp[::factor]

    # Release memory of the originals
    del ppg, abp, data

    # Save the downsampled signals with new fs
    np.savez_compressed(
        output_path,
        ppg=ppg_ds,
        abp=abp_ds,
        fs=float(target_fs)
    )
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Downsample PPG/ABP from 500 Hz to 50 Hz in a .npz file."
    )
    parser.add_argument(
        "input_npz",
        help="Path to the cleaned .npz (containing 'ppg', 'abp', and 'fs=500')."
    )
    parser.add_argument(
        "output_npz",
        help="Path where the downsampled .npz (fs=50) will be saved."
    )
    args = parser.parse_args()
    downsample_to_50hz(args.input_npz, args.output_npz)
    print(f"Saved downsampled signals to '{args.output_npz}'.")


#chmod +x downsample_to_50hz.py
#./downsample_to_50hz.py cleaned_file.npz downsampled_file.npz
