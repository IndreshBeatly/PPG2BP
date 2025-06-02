#!/usr/bin/env python3
"""
full_preprocess_vitaldb.py

This script implements the complete PPG2BP-Net preprocessing pipeline, plus a few “best practice”
improvements (NaN interpolation, optional PPG bandpass). Running it produces train/val/test folders
with per-subject .npz files that contain:
   - 50–100 valid, normalized 10 s PPG segments @ 50 Hz
   - Corresponding SBP/DBP labels (mean of segment)
   - Calibration SBP/DBP (from the first valid segment ≥ 20 min into recording)
   - SDS_SBP, SDS_DBP
   - Demographics (age, sex, weight, height)

Usage:
    python full_preprocess_vitaldb.py \
        --raw_dir raw_data \
        --meta_csv metadata.csv \
        --out_dir processed_data \
        --min_duration_min 10 \
        --fs_target 50 \
        [--bandpass_ppg] 
"""

import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, decimate, find_peaks

# -------------------------
#  Utility Functions
# -------------------------

def interpolate_nans_float32(signal: np.ndarray) -> np.ndarray:
    """
    Replace NaNs by linear interpolation (kept in float32 for memory efficiency).
    If all values are NaN, returns the array unchanged (all NaNs).
    """
    sig = signal.astype(np.float32)
    nans = np.isnan(sig)
    if np.all(nans):
        return sig
    idx = np.arange(len(sig))
    good = ~nans
    interp_vals = np.interp(idx[nans], idx[good], sig[good]).astype(np.float32)
    sig[nans] = interp_vals
    return sig

def butter_lowpass_filter(x: np.ndarray, fs: float, cutoff: float = 25.0, order: int = 4) -> np.ndarray:
    """
    Zero-phase low-pass Butterworth filter (cutoff in Hz) to remove high-frequency noise
    above ~25 Hz (which is well above the PPG pulse frequency).
    """
    nyq = 0.5 * fs
    wn = cutoff / nyq
    b, a = butter(order, wn, btype="lowpass")
    return filtfilt(b, a, x).astype(np.float32)

def butter_bandpass_filter(x: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 8.0, order: int = 2) -> np.ndarray:
    """
    Zero-phase bandpass Butterworth filter from lowcut–highcut (Hz). Useful to remove
    baseline wander (<0.5 Hz) and super-high noise (>8 Hz). Returns float32.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x).astype(np.float32)

def segment_and_clean(ppg_ds: np.ndarray,
                      abp_ds: np.ndarray,
                      fs_ds: float,
                      do_peakfinder: bool = False) -> list[tuple[np.ndarray, float, float]]:
    """
    Break down ppg_ds/abp_ds into nonoverlapping 10 s windows at fs_ds (e.g. 50 Hz).
    Only consider windows whose start >= 20 minutes for *calibration* and *target*.
    Returns a list of (ppg_norm, SBP, DBP) in chronological order, but only from t >= 20 min.
    """
    valid = []
    win_len = int(10 * fs_ds)      # e.g. 10 * 50 = 500 samples
    total_samples = len(ppg_ds)
    num_wins     = total_samples // win_len

    # Compute the window index corresponding to 20 minutes:
    #   20 min = 1200 s → at 50 Hz, 1200 s = 60,000 samples. 60,000 // 500 = 120 windows.
    min_cal_window_index = (20 * 60 * int(fs_ds)) // win_len  # e.g. 120

    for w in range(num_wins):
        if w < min_cal_window_index:
            # Skip any window that starts before 20 minutes.
            continue

        start = w * win_len
        end   = start + win_len
        ppg_win = ppg_ds[start:end]
        abp_win = abp_ds[start:end]

        # T3: must not contain NaN and not be all zeros
        if np.isnan(ppg_win).any() or np.isnan(abp_win).any():
            continue
        if np.all(ppg_win == 0) or np.all(abp_win == 0):
            continue

        # T4: compute SBP & DBP from abp_win
        if do_peakfinder:
            peaks, _ = find_peaks(abp_win, distance=int(0.5 * fs_ds), height=30)
            if len(peaks) > 0:
                SBP_win = float(np.mean(abp_win[peaks]))
            else:
                SBP_win = float(np.max(abp_win))
            troughs, _ = find_peaks(-abp_win, distance=int(0.5 * fs_ds), height=-80)
            if len(troughs) > 0:
                DBP_win = float(np.mean(abp_win[troughs]))
            else:
                DBP_win = float(np.min(abp_win))
        else:
            SBP_win = float(np.max(abp_win))
            DBP_win = float(np.min(abp_win))

        if SBP_win < 70 or SBP_win > 180:
            continue
        if DBP_win < 40 or DBP_win > 110:
            continue

        # Normalize PPG window to zero-mean, unit-variance
        mu = float(np.mean(ppg_win))
        sigma = float(np.std(ppg_win))
        if sigma < 1e-6:
            continue
        ppg_norm = ((ppg_win - mu) / sigma).astype(np.float32)

        valid.append((ppg_norm, SBP_win, DBP_win))

    return valid

def compute_SDS(segments: list[tuple[np.ndarray,float,float]]) -> tuple[float,float]:
    """
    Given a list of (PPG_norm, SBP, DBP) for one subject—chronological order—compute:
      SDS_SBP = std over (SBP_i - SBP_cal) for i = 0..K-1,
      SDS_DBP = std over (DBP_i - DBP_cal),
    where the *first* segment (index=0) is taken as calibration.
    """
    K = len(segments)
    SBP_vals = np.array([seg[1] for seg in segments], dtype=np.float32)
    DBP_vals = np.array([seg[2] for seg in segments], dtype=np.float32)
    if K <= 1:
        return 0.0, 0.0
    SBP_cal = SBP_vals[0]
    DBP_cal = DBP_vals[0]
    delta_SBP = SBP_vals - SBP_cal
    delta_DBP = DBP_vals - DBP_cal
    # Use ddof=1 for sample standard deviation (as paper implies)
    SDS_SBP = float(np.std(delta_SBP, ddof=1))
    SDS_DBP = float(np.std(delta_DBP, ddof=1))
    return SDS_SBP, SDS_DBP

# -------------------------
#  Main Preprocessing
# -------------------------

def full_preprocess(raw_dir: str,
                    meta_csv: str,
                    out_dir: str,
                    min_duration_min: float = 10.0,
                    fs_target: int = 50,
                    do_bandpass_ppg: bool = False):
    """
    1) Read metadata; apply T1 (age/weight/height).
    2) For each remaining subject (caseid):
         a) Load raw PPG/ABP/FS from .npz
         b) T2: skip if fs != 500 or duration < min_duration_min
         c) Interpolate NaNs (PPG & ABP)
         d) (Optional) Bandpass-filter PPG (0.5–8 Hz) to remove wander
         e) Lowpass-filter at 25 Hz (for both PPG & ABP)
         f) Decimate both to fs_target (e.g. 50 Hz)
         g) T3+T4: segment & clean into non-overlapping 10 s windows
         h) If fewer than 50 valid windows, drop subject (T5). If >100, randomly sample 100
         i) Compute SDS for that subject
    3) Build DataFrame of surviving subjects (caseid, demographics, num_segments, SDS)
    4) Random‐shuffle (seed=42) & split into 70/10/20 subjects for train/val/test
    5) For each split, save per‐subject `.npz` under out_dir/{train, val, test}/<caseid>.npz
    """
    np.random.seed(42)
    random.seed(42)

    # 1) Load metadata & apply T1
    meta = pd.read_csv(meta_csv)
    # Keep only those with age ∈ [18,90], weight ∈ [10,100], height ∈ [100,200]
    meta = meta[
        (meta.age.between(18, 90)) &
        (meta.weight.between(10, 100)) &
        (meta.height.between(100, 200))
    ].copy()
    # Ensure caseid is integer
    meta.caseid = meta.caseid.astype(int)

    # 2) Loop over each candidate subject for T2–T5
    balanced_segments = {}  # { caseid: list of (ppg_norm, SBP, DBP) }
    sds_dict = {}           # { caseid: (SDS_SBP, SDS_DBP) }

    dropped_t2 = 0  # missing signals, fs != 500, too short
    dropped_t3 = 0  # no valid segments after cleaning
    dropped_t5 = 0  # <50 segments

    for idx, row in meta.iterrows():
        cid = int(row.caseid)
        raw_path = os.path.join(raw_dir, str(cid), "signals.npz")
        if not os.path.isfile(raw_path):
            dropped_t2 += 1
            continue

        data = np.load(raw_path)
        # a) Load raw PPG & ABP
        raw_ppg = data.get("ppg", None)
        raw_abp = data.get("abp", None)
        fs_raw = float(data.get("fs", 0.0))

        if raw_ppg is None or raw_abp is None or fs_raw != 500.0:
            dropped_t2 += 1
            continue

        total_samples = min(len(raw_ppg), len(raw_abp))
        # b) T2: min duration
        if total_samples < int(500 * 60 * min_duration_min):
            dropped_t2 += 1
            continue

        # Clip to same length
        raw_ppg = raw_ppg[:total_samples]
        raw_abp = raw_abp[:total_samples]

        # c) Interpolate NaNs
        ppg_interp = interpolate_nans_float32(raw_ppg)
        abp_interp = interpolate_nans_float32(raw_abp)

        # d) Optional: Bandpass-filter PPG (0.5–8 Hz) to remove baseline wander
        if do_bandpass_ppg:
            ppg_interp = butter_bandpass_filter(ppg_interp, fs_raw, lowcut=0.5, highcut=8.0, order=2)

        # e) Lowpass both PPG and ABP at 25 Hz
        ppg_low = butter_lowpass_filter(ppg_interp, fs_raw, cutoff=25.0, order=4)
        abp_low = butter_lowpass_filter(abp_interp, fs_raw, cutoff=25.0, order=4)

        # f) Decimate to fs_target (e.g. 50 Hz)
        decim_factor = int(fs_raw // fs_target)
        if decim_factor < 1:
            dropped_t2 += 1
            continue

        ppg_ds = decimate(ppg_low, decim_factor, ftype="iir", zero_phase=True).astype(np.float32)
        abp_ds = decimate(abp_low, decim_factor, ftype="iir", zero_phase=True).astype(np.float32)

        # g) T3+T4: segment & clean into non-overlapping 10 s windows
        clean_segs = segment_and_clean(ppg_ds, abp_ds, fs_target, do_peakfinder=False)
        if len(clean_segs) < 50:
            dropped_t5 += 1
            continue

        # h) T5: balance segments per subject (50–100)
        K = len(clean_segs)
        if K < 50:
            dropped_t5 += 1
            continue
        if K > 100:
            sampled = random.sample(clean_segs, 100)
        else:
            sampled = clean_segs

        balanced_segments[cid] = sampled

        # i) Compute SDS (subject-calibration SD)
        SDS_SBP, SDS_DBP = compute_SDS(sampled)
        sds_dict[cid] = (SDS_SBP, SDS_DBP)

    print("== Preprocessing Summary ==")
    print(f"Subjects dropped at T2 (missing/fs!=500/too short): {dropped_t2}")
    print(f"Subjects dropped at T3 (no valid segments): {dropped_t3}")
    print(f"Subjects dropped at T5 (<50 segments): {dropped_t5}")
    print(f"Subjects remaining: {len(balanced_segments)}  (should be ≈ 4185)\n")

    # 3) Build DataFrame of all surviving subjects
    rows = []
    for cid, segs in balanced_segments.items():
        age = float(meta.loc[meta.caseid == cid, "age"].iloc[0])
        sex = meta.loc[meta.caseid == cid, "sex"].iloc[0]
        weight = float(meta.loc[meta.caseid == cid, "weight"].iloc[0])
        height = float(meta.loc[meta.caseid == cid, "height"].iloc[0])
        num_segs = len(segs)
        SDS_SBP, SDS_DBP = sds_dict[cid]
        rows.append({
            "caseid": cid,
            "age": age,
            "sex": sex,
            "weight": weight,
            "height": height,
            "num_segments": num_segs,
            "SDS_SBP": SDS_SBP,
            "SDS_DBP": SDS_DBP
        })
    df_all = pd.DataFrame(rows)
    df_all = df_all.sort_values("caseid").reset_index(drop=True)

    # 4) Subject-independent 70/10/20 split (seed=42)
    all_cids = df_all.caseid.tolist()
    random.shuffle(all_cids)
    n_total = len(all_cids)
    n_train = int(0.7 * n_total)  # 70% for training
    n_val   = int(0.1 * n_total)  # 10% for validation
    n_test  = n_total - n_train - n_val  # should be 788

    train_cids = all_cids[:n_train]
    val_cids   = all_cids[n_train:n_train+n_val]
    test_cids  = all_cids[n_train+n_val:n_train+n_val+n_test]

    assert len(train_cids) == n_train
    assert len(val_cids)   == n_val
    assert len(test_cids)  == n_test

    # 5) Save per-subject .npz for each split
    for split, cids in [("train", train_cids), ("val", val_cids), ("test", test_cids)]:
        split_dir = os.path.join(out_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for cid in cids:
            segs = balanced_segments[cid]
            # Stack them into arrays
            PPGs = np.stack([s[0] for s in segs], axis=0)
            SBPs = np.array([s[1] for s in segs], dtype=np.float32)
            DBPs = np.array([s[2] for s in segs], dtype=np.float32)
            SDS_SBP, SDS_DBP = sds_dict[cid]

            # Demographics
            age = float(meta.loc[meta.caseid == cid, "age"].iloc[0])
            sex = meta.loc[meta.caseid == cid, "sex"].iloc[0]
            weight = float(meta.loc[meta.caseid == cid, "weight"].iloc[0])
            height = float(meta.loc[meta.caseid == cid, "height"].iloc[0])

             # Define SBP_cal and DBP_cal from the first valid segment
            SBP_cal = segs[0][1]   # the SBP label of window index 0
            DBP_cal = segs[0][2]   # the DBP label of window index 0
            
            np.savez_compressed(
                os.path.join(split_dir, f"{cid}.npz"),
                PPG_segments=PPGs,   # float32 array shape (K, win_len)
                SBP_labels=SBPs,     # float32 array shape (K,)
                DBP_labels=DBPs,  
                SBP_cal=np.float32(SBP_cal),
                DBP_cal=np.float32(DBP_cal),
                SDS_SBP=np.float32(SDS_SBP),
                SDS_DBP=np.float32(SDS_DBP),
                age=np.float32(age),
                sex=sex,
                weight=np.float32(weight),
                height=np.float32(height)
               # the DBP label of window index 0
            )

    # (Optional) Save summary CSVs
    os.makedirs(out_dir, exist_ok=True)
    df_all.to_csv(os.path.join(out_dir, "all_subjects_info.csv"), index=False)
    pd.DataFrame({"train_cid": train_cids}).to_csv(os.path.join(out_dir, "train_cids.csv"), index=False)
    pd.DataFrame({"val_cid":   val_cids}).to_csv(os.path.join(out_dir, "val_cids.csv"), index=False)
    pd.DataFrame({"test_cid":  test_cids}).to_csv(os.path.join(out_dir, "test_cids.csv"), index=False)

    print("=== Finished preprocessing. Output directory:", out_dir, "===\n")
    print("Subject counts: train =", len(train_cids),
          " val =", len(val_cids),
          " test =", len(test_cids))
    print("Average SDS_SBP (train) =", df_all[df_all.caseid.isin(train_cids)]["SDS_SBP"].mean())
    print("Average SDS_SBP (val)   =", df_all[df_all.caseid.isin(val_cids)]["SDS_SBP"].mean())
    print("Average SDS_SBP (test)  =", df_all[df_all.caseid.isin(test_cids)]["SDS_SBP"].mean())


# -------------------------
#  Entry Point
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full PPG2BP-Net preprocessing (T1–T5, SDS, splits).")
    parser.add_argument("--raw_dir",  required=True,
                        help="Root folder containing raw_data/<caseid>/signals.npz")
    parser.add_argument("--meta_csv", required=True,
                        help="Path to metadata CSV (with columns: caseid, age, sex, weight, height, etc.)")
    parser.add_argument("--out_dir",  required=True,
                        help="Output folder; will contain train/, val/, test/ subfolders.")
    parser.add_argument("--min_duration_min", type=float, default=10.0,
                        help="Minimum recording length (minutes) to keep a subject (default=10).")
    parser.add_argument("--fs_target", type=int, default=50,
                        help="Target downsampled frequency in Hz (default=50).")
    parser.add_argument("--bandpass_ppg", action="store_true",
                        help="If set, bandpass-filter PPG from 0.5–8 Hz before lowpass & decimate.")
    args = parser.parse_args()

    full_preprocess(
        raw_dir=args.raw_dir,
        meta_csv=args.meta_csv,
        out_dir=args.out_dir,
        min_duration_min=args.min_duration_min,
        fs_target=args.fs_target,
        do_bandpass_ppg=args.bandpass_ppg
    )
