import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class PPG2BPDataset(Dataset):
    """
    PyTorch Dataset for PPG2BP-Net, using the “6th window as calibration” rule.

    Args:
      split_txt:     Path to a text file (one subject_id per line), e.g. 'data/data_split/split_train.txt'
      data_root:     Root folder where per‐subject .npz files live (data/final)
                     Each subject has: data_root/<subject_id>/signals_with_metadata.npz
      min_windows:   Minimum number of windows; we require at least 6 windows per subject
                     (otherwise there is no “6th window” to use as calibration).
    """
    def __init__(self, split_txt: str, data_root: str = "data/final", min_windows: int = 6):
        super().__init__()

        self.data_root = data_root
        self.min_windows = min_windows

        # 1) Read the split file to get a list of subject_ids
        self.subject_ids = []
        with open(split_txt, 'r') as f:
            for line in f:
                sid = line.strip()
                if len(sid) == 0:
                    continue
                self.subject_ids.append(sid)

        # 2) Build an index mapping: for each subject, we'll store (subject_id, number_of_windows)
        #    But actually, we need to flatten all the “non-calibration” windows across all subjects
        #    into a single list of (subject_id, window_index) pairs. Then __len__ = total_samples.
        self.index_map = []  # will hold tuples (subject_id, window_index)
        for sid in self.subject_ids:
            npz_path = os.path.join(self.data_root, sid, "signals_with_metadata.npz")
            if not os.path.isfile(npz_path):
                continue

            with np.load(npz_path) as data:
                ppg_windows = data["ppg_windows"]      # shape (N, 500)
                sbp_vals    = data["sbp_values"]       # shape (N,)
                dbp_vals    = data["dbp_values"]       # shape (N,)
                # (fs = data["fs"] but we assume it's correct 50)

                N = ppg_windows.shape[0]
                if N < self.min_windows:
                    # Skip subjects with fewer than min_windows (i.e. <6 windows)
                    continue

                # The 6th window index = 5 is reserved as calibration.
                # All other indices (0–4, 6–N−1) will be training samples.
                for wi in range(N):
                    if wi == 5:
                        # skip calibration index itself
                        continue
                    # store a reference to this sample
                    self.index_map.append((sid, wi))

        # At this point, index_map length = sum_over_subjects(N_for_that_subject − 1),
        # for all subjects who have at least 6 windows.
        self.total_samples = len(self.index_map)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Returns one training sample as a dictionary with these keys:
          - 'ppg_target':  Tensor of shape (1, 500)  (float32)
          - 'ppg_calib':   Tensor of shape (1, 500)  (float32)
          - 'sbp_calib':   Tensor of shape (1,)      (float32)
          - 'dbp_calib':   Tensor of shape (1,)      (float32)
          - 'sbp_true':    Tensor of shape (1,)      (float32)
          - 'dbp_true':    Tensor of shape (1,)      (float32)
        """
        sid, win_idx = self.index_map[idx]
        npz_path = os.path.join(self.data_root, sid, "signals_with_metadata.npz")
        with np.load(npz_path) as data:
            # Load all windows, then pick out indices
            ppg_windows = data["ppg_windows"]      # (N, 500)
            sbp_vals    = data["sbp_values"]       # (N,)
            dbp_vals    = data["dbp_values"]       # (N,)

            # Calibration index is always 5:
            ppg_calib = ppg_windows[5]             # shape (500,)
            sbp_calib = sbp_vals[5]                # scalar
            dbp_calib = dbp_vals[5]                # scalar

            # Target window:
            ppg_target = ppg_windows[win_idx]      # shape (500,)
            sbp_true   = sbp_vals[win_idx]         # scalar
            dbp_true   = dbp_vals[win_idx]         # scalar

        # Convert everything to float32 tensors, with appropriate dims.
        sample = {
            "ppg_target": torch.from_numpy(ppg_target.astype(np.float32)).unsqueeze(0),  # (1, 500)
            "ppg_calib":  torch.from_numpy(ppg_calib.astype(np.float32)).unsqueeze(0),   # (1, 500)
            "sbp_calib":  torch.tensor([sbp_calib], dtype=torch.float32),               # (1,)
            "dbp_calib":  torch.tensor([dbp_calib], dtype=torch.float32),               # (1,)
            "sbp_true":   torch.tensor([sbp_true], dtype=torch.float32),                # (1,)
            "dbp_true":   torch.tensor([dbp_true], dtype=torch.float32),                # (1,)
        }
        return sample

# ----------------------------
# Example: Creating DataLoaders
# ----------------------------
if __name__ == "__main__":
    train_dataset = PPG2BPDataset(split_txt="data/data_split/split_train.txt", data_root="data/final")
    val_dataset   = PPG2BPDataset(split_txt="data/data_split/split_val.txt",   data_root="data/final")
    test_dataset  = PPG2BPDataset(split_txt="data/data_split/split_test.txt",  data_root="data/final")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    print(f"#train samples: {len(train_dataset)}")
    print(f"#val   samples: {len(val_dataset)}")
    print(f"#test  samples: {len(test_dataset)}")

    # Fetch one batch to verify shapes:
    batch = next(iter(train_loader))
    print({k: v.shape for k,v in batch.items()})
    # Should print:
    # {
    #   'ppg_target': (64, 1, 500),
    #   'ppg_calib':  (64, 1, 500),
    #   'sbp_calib':  (64, 1),
    #   'dbp_calib':  (64, 1),
    #   'sbp_true':   (64, 1),
    #   'dbp_true':   (64, 1)
    # }
