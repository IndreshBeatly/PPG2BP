#!/usr/bin/env python3
import os
import argparse
import shutil                     # ← Added
import numpy as np
import pandas as pd

from modules.strip_time            import strip_times
from modules.nan_removal           import remove_nans
from modules.decimation50hz        import downsample_to_50hz
from modules.tensec_window_split   import select_top_windows_with_snr
from modules.bp_labels             import sbp_dbp_labels
from modules.normalise             import filter_ppg_file  # band‐pass + normalize

INPUT_ROOT    = "data/raw_api"
OUTPUT_ROOT   = "data/final"
TMP_ROOT      = "tmp_pipeline"
METADATA_CSV  = "data/metadata/vitaldb_metadata.csv"  # adjust as needed

# Ensure output directories exist
os.makedirs(OUTPUT_ROOT, exist_ok=True)
os.makedirs(TMP_ROOT, exist_ok=True)


def load_metadata(csv_path: str) -> pd.DataFrame:
    """
    Read the metadata CSV into a DataFrame, indexed by 'caseid' (string).
    We assume the CSV column 'caseid' matches the patient_id folder names exactly.
    """
    df = pd.read_csv(csv_path, dtype={"caseid": str})
    df = df.set_index("caseid", drop=False)
    return df


def run_pipeline_for_patient(patient_id: str, metadata_df: pd.DataFrame):
    """
    Run the full pipeline steps 1–6 for one patient. If Step 4
    (select_top_windows_with_snr) returns False → drop patient.
    Otherwise, merge in metadata (height, weight, age, gender) and save final .npz.
    Once the final .npz is saved, delete the temporary folder to save space.
    """
    input_dir   = os.path.join(INPUT_ROOT, patient_id)
    signal_file = os.path.join(input_dir, "signals.npz")

    if not os.path.isfile(signal_file):
        print(f"[SKIP] Missing signals.npz for patient {patient_id}")
        return False

    # Create a temp subfolder for this patient
    tmp_dir = os.path.join(TMP_ROOT, patient_id)
    os.makedirs(tmp_dir, exist_ok=True)

    # Intermediate file paths (all under tmp_dir)
    path_strip  = os.path.join(tmp_dir, "step1_strip.npz")
    path_nans   = os.path.join(tmp_dir, "step2_nans.npz")
    path_ds50   = os.path.join(tmp_dir, "step3_ds50.npz")
    path_snr    = os.path.join(tmp_dir, "step4_snr.npz")
    path_labels = os.path.join(tmp_dir, "step5_labels.npz")
    path_norm   = os.path.join(tmp_dir, "step6_filtered_norm.npz")

    try:
        print(f"[INFO] Processing patient {patient_id}")

        # Step 1: Strip head/tail
        print("  ➤ Step 1: Stripping start/end time…")
        strip_times(signal_file, path_strip)

        # Step 2: Remove NaNs
        print("  ➤ Step 2: Removing NaNs…")
        remove_nans(path_strip, path_nans)

        # Step 3: Downsample to 50 Hz
        print("  ➤ Step 3: Downsampling to 50 Hz…")
        downsample_to_50hz(path_nans, path_ds50)

        # Step 4: Select top SNR windows (requires ≥50 valid windows)
        print("  ➤ Step 4: Selecting top SNR windows…")
        has_windows = select_top_windows_with_snr(path_ds50, path_snr, min_windows=50)
        if not has_windows:
            print(f"[SKIP] Patient {patient_id}: fewer than 50 valid 10 s windows → dropped\n")
            return False

        # Step 5: Compute SBP/DBP labels
        print("  ➤ Step 5: Computing SBP/DBP labels…")
        sbp_dbp_labels(path_snr, path_labels)

        # Step 6: Bandpass + Normalize PPG windows
        print("  ➤ Step 6: Bandpass + Normalize PPG windows…")
        filter_ppg_file(path_labels, path_norm)

        # Load final‐stage data (after filtering + normalization)
        with np.load(path_norm) as final_data:
            ppg_windows = final_data["ppg_windows"]
            sbp_values  = final_data["sbp_values"]
            dbp_values  = final_data["dbp_values"]
            fs          = final_data["fs"]

        # ALSO load SNR/ABP windows from step 4 (optional, if you want to store)
        with np.load(path_snr) as snr_data:
            snr_scores  = snr_data["snr_scores"]
            abp_windows = snr_data["abp_windows"]

        # Look up metadata for this patient_id
        if patient_id not in metadata_df.index:
            raise KeyError(f"Patient {patient_id} not found in metadata CSV")
        row = metadata_df.loc[patient_id]

        # Example metadata columns; adjust if your CSV uses different names
        height = float(row.get("height", np.nan))
        weight = float(row.get("weight", np.nan))
        age    = float(row.get("age", np.nan))
        gender = str(row.get("sex", ""))
        bmi    = float(row.get("bmi", np.nan))

        # Prepare final output directory
        patient_out_dir = os.path.join(OUTPUT_ROOT, patient_id)
        os.makedirs(patient_out_dir, exist_ok=True)
        final_output = os.path.join(patient_out_dir, "signals_with_metadata.npz")

        # Save everything: windows, labels, SNR/ABP, plus metadata
        np.savez_compressed(
            final_output,
            # Filtered & normalized PPG windows
            ppg_windows=ppg_windows,
            # SBP/DBP labels
            sbp_values=sbp_values,
            dbp_values=dbp_values,
            # (Optional) SNR scores and ABP windows
            snr_scores=snr_scores,
            abp_windows=abp_windows,
            # Sampling frequency
            fs=fs,
            # Now the metadata fields:
            height=np.array([height], dtype=np.float32),
            weight=np.array([weight], dtype=np.float32),
            age=np.array([age], dtype=np.float32),
            bmi=np.array([bmi], dtype=np.float32),
            gender=np.array([gender])
        )

        # ────────────────── CLEAN UP INTERMEDIATE FILES ──────────────────
        # Remove the entire temporary directory for this patient now that final_output is saved
        try:
            shutil.rmtree(tmp_dir)
        except Exception as rm_err:
            print(f"[WARNING] Could not remove temp folder {tmp_dir}: {rm_err}")

        print(f"[DONE] Patient {patient_id} → saved to '{final_output}' (temp files deleted)\n")
        return True

    except Exception as e:
        print(f"[ERROR] Patient {patient_id}: {e}\n")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Process all patients and append metadata into final .npz."
    )
    parser.add_argument(
        "--raw_dir", default=INPUT_ROOT,
        help="Folder containing per-patient subfolders, each with signals.npz."
    )
    parser.add_argument(
        "--out_dir", default=OUTPUT_ROOT,
        help="Folder where final per-patient folders + .npz will be saved."
    )
    parser.add_argument(
        "--meta_csv", default=METADATA_CSV,
        help="CSV file containing patient metadata (must have 'caseid' column)."
    )
    args = parser.parse_args()

    # Load metadata CSV once
    metadata_df = load_metadata(args.meta_csv)

    # Get list of patient IDs (folder names) under raw_dir
    patient_ids = sorted([
        pid for pid in os.listdir(args.raw_dir)
        if os.path.isdir(os.path.join(args.raw_dir, pid))
    ], key=lambda x: int(x) if x.isdigit() else x)

    print(f"[INFO] Found {len(patient_ids)} patients in '{args.raw_dir}'.\n")

    passed_ids = []
    for pid in patient_ids:
        success = run_pipeline_for_patient(pid, metadata_df)
        if success:
            passed_ids.append(pid)

    print(f"[INFO] {len(passed_ids)} patients passed all checks: {passed_ids}")


if __name__ == "__main__":
    main()
