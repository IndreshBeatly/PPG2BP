import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# Load case IDs from CSV files
train_cids = pd.read_csv("processed_data/train_cids.csv")["train_cid"].tolist()
val_cids = pd.read_csv("processed_data/val_cids.csv")["val_cid"].tolist()
test_cids = pd.read_csv("processed_data/test_cids.csv")["test_cid"].tolist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# (Include all the classes & functions defined above: PPG2BP_Dataset, sample_train_batch,
#  OneDCNNBranch, PPG2BP_Net, train(), evaluate_testset, etc.)

class PPG2BP_Dataset(Dataset):
    def __init__(self, data_dir, caseids):
        """
        data_dir: e.g. "processed_data/train/"
        caseids:  list of int case IDs (e.g. [1,4,7,10,13,16,17])
        """
        self.data_dir = data_dir
        self.caseids = caseids

    def __len__(self):
        # The "length" is the number of subjects available. Actual batch size is fixed to 1 subject per index.
        return len(self.caseids)

    def __getitem__(self, idx):
        """
        Returns a single subject's entire data,
        so the DataLoader can sample 64 distinct subjects per batch.
        We'll collate them manually in the training loop.
        """
        cid = self.caseids[idx]
        fname = os.path.join(self.data_dir, f"{cid}.npz")
        data = np.load(fname)
        PPGs = data["PPG_segments"]   # shape (K, 500), dtype=float32
        SBPs = data["SBP_labels"]     # shape (K,), dtype=float32
        DBPs = data["DBP_labels"]     # shape (K,), dtype=float32
        SBP_cal = float(data["SBP_cal"])   # scalar
        DBP_cal = float(data["DBP_cal"])   # scalar

        # Return everything needed for one subject
        return {
            "caseid": cid,
            "PPGs": torch.from_numpy(PPGs),       # (K, 500)
            "SBPs": torch.from_numpy(SBPs),       # (K,)
            "DBPs": torch.from_numpy(DBPs),       # (K,)
            "SBP_cal": torch.tensor(SBP_cal),     # scalar tensor
            "DBP_cal": torch.tensor(DBP_cal)      # scalar tensor
        }

def sample_train_batch(dataset, batch_size=64):
    """
    dataset: an instance of PPG2BP_Dataset for `train/`
    batch_size: number of distinct subjects to sample
    Returns:
      - ppg_cal_B:   (batch_size, 500) tensor
      - bp_cal_B:    (batch_size, 2)   tensor [SBP_cal, DBP_cal]
      - ppg_targ_B:  (batch_size, 500) tensor
      - bp_targ_B:   (batch_size, 2)   tensor [SBP_targ, DBP_targ]
    """
    selected_indices = random.sample(range(len(dataset)), batch_size)
    ppg_cal_list   = []
    bp_cal_list    = []
    ppg_targ_list  = []
    bp_targ_list   = []

    for idx in selected_indices:
        entry = dataset[idx]
        PPGs = entry["PPGs"]        # shape (K, 500)
        SBPs = entry["SBPs"]        # shape (K,)
        DBPs = entry["DBPs"]        # shape (K,)
        SBP_cal = entry["SBP_cal"]  # scalar tensor
        DBP_cal = entry["DBP_cal"]  # scalar tensor

        # Always pick index=0 as calibration
        ppg_cal_list.append(PPGs[0])                # (500,)
        bp_cal_list.append(torch.stack([SBP_cal, DBP_cal]))  # (2,)

        # Pick a random target index in [1..K-1]
        K = PPGs.shape[0]
        if K <= 1:
            # Should never happen if T5 was enforced, but just in case:
            targ_idx = 0
        else:
            targ_idx = random.randint(1, K-1)
        ppg_targ_list.append(PPGs[targ_idx])        # (500,)
        bp_targ_list.append(torch.stack([SBPs[targ_idx], DBPs[targ_idx]]))  # (2,)

    ppg_cal_B  = torch.stack(ppg_cal_list, dim=0)  # shape (batch_size, 500)
    bp_cal_B   = torch.stack(bp_cal_list, dim=0)   # shape (batch_size, 2)
    ppg_targ_B = torch.stack(ppg_targ_list, dim=0) # shape (batch_size, 500)
    bp_targ_B  = torch.stack(bp_targ_list, dim=0)  # shape (batch_size, 2)

    return ppg_cal_B, bp_cal_B, ppg_targ_B, bp_targ_B
class OneDCNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4   = nn.BatchNorm1d(256)
        self.pool  = nn.AvgPool1d(kernel_size=2)       # reduces length from 500 → 250
        self.drop  = nn.Dropout(0.3)
        # After conv+pool, each feature map is (batch, 256, 250) → flatten to 256*250
        self.fc    = nn.Linear(256 * 250, 8)
        self.bn_fc = nn.BatchNorm1d(8)

    def forward(self, x):
        # x: (batch_size, 1, 500)
        x = F.relu(self.bn1(self.conv1(x)))  # → (batch, 32, 500)
        x = F.relu(self.bn2(self.conv2(x)))  # → (batch, 64, 500)
        x = F.relu(self.bn3(self.conv3(x)))  # → (batch,128, 500)
        x = F.relu(self.bn4(self.conv4(x)))  # → (batch,256, 500)
        x = self.pool(x)                     # → (batch,256, 250)
        x = self.drop(x)
        b, c, t = x.shape
        x = x.view(b, c * t)                 # → (batch, 256*250)
        x = F.relu(self.bn_fc(self.fc(x)))   # → (batch, 8)
        return x                             # final 8-D feature vector

class PPG2BP_Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Two identical CNN branches (separate weights)
        self.cnn_cal  = OneDCNNBranch()
        self.cnn_targ = OneDCNNBranch()

        # MLP for numeric calibration BP
        self.bp_mlplayer = nn.Sequential(
            nn.Linear(2, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        # Final fusion regressor
        self.fc1 = nn.Linear(8 + 16, 128)  # input = |f_targ - f_cal| (8) + h_cal (16) = 24
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 2)        # outputs (SBP_pred, DBP_pred)

    def forward(self, ppg_cal, bp_cal, ppg_targ):
        """
        ppg_cal: shape (batch_size, 1, 500)
        bp_cal:  shape (batch_size, 2)
        ppg_targ: shape (batch_size, 1, 500)
        """
        f_cal  = self.cnn_cal(ppg_cal)      # → (batch_size, 8)
        f_targ = self.cnn_targ(ppg_targ)    # → (batch_size, 8)
        delta  = torch.abs(f_targ - f_cal)  # → (batch_size, 8)

        h_cal  = self.bp_mlplayer(bp_cal)   # → (batch_size, 16)
        fusion = torch.cat([delta, h_cal], dim=1)  # → (batch_size, 24)

        x = F.relu(self.bn1(self.fc1(fusion)))     # → (batch_size, 128)
        x = F.relu(self.bn2(self.fc2(x)))          # → (batch_size, 64)
        out = self.fc3(x)                          # → (batch_size, 2)
        return out  # [SBP_pred, DBP_pred]
    
def train(model, optimizer, criterion, train_dataset, val_dataset,
          n_epochs=1000, batch_size=64, patience_limit=10):
    """
    model:       PPG2BP_Net instance
    optimizer:   Adam optimizer
    criterion:   MSELoss
    train_dataset: instance of PPG2BP_Dataset (train split)
    val_dataset:   instance of PPG2BP_Dataset (val split)
    n_epochs:    maximum number of epochs
    batch_size:  64 (as in paper)
    patience_limit: 10 epochs without improvement → early stop
    """
    best_val_loss = float("inf")
    patience = 0

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Decide how many batches per epoch: for simplicity, 
        # iterate so that we see each train subject roughly once.
        num_train_subjects = len(train_dataset)
        num_batches_per_epoch = num_train_subjects // batch_size
        if num_batches_per_epoch < 1:
            num_batches_per_epoch = 1

        for _ in range(num_batches_per_epoch):
            # Sample one batch of 64 distinct subjects
            ppg_cal_B, bp_cal_B, ppg_t_B, bp_t_B = sample_train_batch(train_dataset, batch_size)

            # Move to device and reshape PPGs for CNN
            ppg_cal_B = ppg_cal_B.unsqueeze(1).to(device)  # (batch_size, 1, 500)
            bp_cal_B  = bp_cal_B.to(device)                # (batch_size, 2)
            ppg_t_B   = ppg_t_B.unsqueeze(1).to(device)    # (batch_size, 1, 500)
            bp_t_B    = bp_t_B.to(device)                  # (batch_size, 2)

            optimizer.zero_grad()
            preds = model(ppg_cal_B, bp_cal_B, ppg_t_B)    # → (batch_size, 2)
            loss = criterion(preds, bp_t_B)                # averaged over 2 outputs
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / num_batches_per_epoch

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            total_val_windows = 0

            for idx in range(len(val_dataset)):
                entry = val_dataset[idx]
                PPGs = entry["PPGs"].to(device)  # (K, 500)
                SBPs = entry["SBPs"].to(device)  # (K,)
                DBPs = entry["DBPs"].to(device)  # (K,)
                SBP_cal = entry["SBP_cal"].to(device)  # scalar
                DBP_cal = entry["DBP_cal"].to(device)  # scalar

                K = PPGs.shape[0]
                if K <= 2:
                    continue  # skip if fewer than 3 windows

                # Build calibration feature by averaging first two windows
                ppg_cal_01 = PPGs[:2, :].unsqueeze(1)  # (2,1,500)
                f_cal_1 = model.cnn_cal(ppg_cal_01[0:1, :, :])  # (1,8)
                f_cal_2 = model.cnn_cal(ppg_cal_01[1:2, :, :])  # (1,8)
                f_cal = 0.5 * (f_cal_1 + f_cal_2)                # (1,8)

                bp_cal = torch.stack([SBP_cal, DBP_cal]).unsqueeze(0)  # (1,2)
                h_cal = model.bp_mlplayer(bp_cal)                      # (1,16)

                # Target windows: indices 2…K-1
                num_targets = K - 2
                ppg_targets = PPGs[2:K, :].unsqueeze(1)  # (K-2,1,500)
                sbp_targets = SBPs[2:K].unsqueeze(1)     # (K-2,1)
                dbp_targets = DBPs[2:K].unsqueeze(1)     # (K-2,1)
                bp_targets = torch.cat([sbp_targets, dbp_targets], dim=1)  # (K-2, 2)

                f_targs = model.cnn_targ(ppg_targets)       # (K-2, 8)
                f_cal_rep = f_cal.repeat(num_targets, 1)    # (K-2, 8)
                delta = torch.abs(f_targs - f_cal_rep)      # (K-2, 8)
                h_cal_rep = h_cal.repeat(num_targets, 1)    # (K-2,16)
                fusion = torch.cat([delta, h_cal_rep], dim=1)  # (K-2,24)

                x = F.relu(model.bn1(model.fc1(fusion)))  # (K-2,128)
                x = F.relu(model.bn2(model.fc2(x)))       # (K-2,64)
                preds_val = model.fc3(x)                  # (K-2,2)

                val_loss += criterion(preds_val, bp_targets).item() * (num_targets)
                total_val_windows += num_targets

            avg_val_loss = val_loss / total_val_windows if total_val_windows > 0 else float("inf")

        print(f"Epoch {epoch} → Train Loss: {avg_epoch_loss:.4f}   Val Loss: {avg_val_loss:.4f}")

        # Early Stopping Check
        if avg_val_loss + 1e-4 < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_ppg2bpnet.pth")
            patience = 0
            print("  ** New best model saved. **")
        else:
            patience += 1
            if patience >= patience_limit:
                print("Early stopping triggered (no improvement for {} epochs).".format(patience_limit))
                break

# After this, "best_ppg2bpnet.pth" holds the best weights.
def evaluate_testset(model, checkpoint_path, test_dataset):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_errors_SBP = []
    all_errors_DBP = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            entry = test_dataset[idx]
            PPGs = entry["PPGs"].to(device)       # (K, 500)
            SBPs = entry["SBPs"].to(device)       # (K,)
            DBPs = entry["DBPs"].to(device)       # (K,)
            SBP_cal = entry["SBP_cal"].to(device) # scalar
            DBP_cal = entry["DBP_cal"].to(device) # scalar
            cid = entry["caseid"]

            K = PPGs.shape[0]
            if K <= 2:
                continue  # skip if too few windows

            # Build calibration feature (average of first two)
            ppg_cal_01 = PPGs[:2, :].unsqueeze(1)  # (2,1,500)
            f_cal_1 = model.cnn_cal(ppg_cal_01[0:1, :, :])
            f_cal_2 = model.cnn_cal(ppg_cal_01[1:2, :, :])
            f_cal = 0.5 * (f_cal_1 + f_cal_2)  # (1,8)

            bp_cal = torch.stack([SBP_cal, DBP_cal]).unsqueeze(0)  # (1,2)
            h_cal = model.bp_mlplayer(bp_cal)                      # (1,16)

            # Targets: indices 2..K-1
            num_targets = K - 2
            ppg_targets = PPGs[2:K, :].unsqueeze(1)    # (K-2,1,500)
            sbp_targets = SBPs[2:K].cpu().numpy()      # (K-2,)
            dbp_targets = DBPs[2:K].cpu().numpy()      # (K-2,)

            f_targs = model.cnn_targ(ppg_targets)      # (K-2, 8)
            f_cal_rep = f_cal.repeat(num_targets, 1)   # (K-2, 8)
            delta = torch.abs(f_targs - f_cal_rep)     # (K-2, 8)
            h_cal_rep = h_cal.repeat(num_targets, 1)   # (K-2, 16)
            fusion = torch.cat([delta, h_cal_rep], dim=1)  # (K-2, 24)

            x = F.relu(model.bn1(model.fc1(fusion)))   # (K-2, 128)
            x = F.relu(model.bn2(model.fc2(x)))        # (K-2, 64)
            preds = model.fc3(x).cpu().numpy()         # (K-2, 2)

            errs_SBP = preds[:, 0] - sbp_targets       # numpy array (K-2,)
            errs_DBP = preds[:, 1] - dbp_targets

            all_errors_SBP.extend(errs_SBP.tolist())
            all_errors_DBP.extend(errs_DBP.tolist())

    # Convert to numpy
    all_errors_SBP = np.array(all_errors_SBP)
    all_errors_DBP = np.array(all_errors_DBP)

    # ME, SD, MAE
    ME_SBP  = np.mean(all_errors_SBP)
    SD_SBP  = np.std(all_errors_SBP)
    MAE_SBP = np.mean(np.abs(all_errors_SBP))

    ME_DBP  = np.mean(all_errors_DBP)
    SD_DBP  = np.std(all_errors_DBP)
    MAE_DBP = np.mean(np.abs(all_errors_DBP))

    print("Test SBP → ME: {:.3f} mmHg   SD: {:.3f} mmHg   MAE: {:.3f} mmHg".format(ME_SBP, SD_SBP, MAE_SBP))
    print("Test DBP → ME: {:.3f} mmHg   SD: {:.3f} mmHg   MAE: {:.3f} mmHg".format(ME_DBP, SD_DBP, MAE_DBP))

    # BHS grading
    for bound in [5, 10, 15]:
        pct_SBP = np.mean(np.abs(all_errors_SBP) <= bound) * 100
        pct_DBP = np.mean(np.abs(all_errors_DBP) <= bound) * 100
        print(f"% |error| ≤ {bound} mmHg → SBP: {pct_SBP:.1f}%, DBP: {pct_DBP:.1f}%")

    # Check AAMI criteria: |ME| ≤ 5 mmHg, SD ≤ 8 mmHg, n ≥ 85.
    print("\nAAMI Check:")
    print(f"  SBP |ME| = {abs(ME_SBP):.3f} (≤ 5?),   SD = {SD_SBP:.3f} (≤ 8?)")
    print(f"  DBP |ME| = {abs(ME_DBP):.3f} (≤ 5?),   SD = {SD_DBP:.3f} (≤ 8?)")

    return {
        "ME_SBP": ME_SBP, "SD_SBP": SD_SBP, "MAE_SBP": MAE_SBP,
        "ME_DBP": ME_DBP, "SD_DBP": SD_DBP, "MAE_DBP": MAE_DBP
    }


if __name__ == "__main__":
    # 1) Seed
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # 2) Build dataset objects
    train_dataset = PPG2BP_Dataset(data_dir="processed_data/train", caseids=train_cids)
    val_dataset   = PPG2BP_Dataset(data_dir="processed_data/val",   caseids=val_cids)
    test_dataset  = PPG2BP_Dataset(data_dir="processed_data/test",  caseids=test_cids)

    # 3) Instantiate model, optimizer, loss
    model = PPG2BP_Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    # 4) Train until early stopping
    train(model, optimizer, criterion, train_dataset, val_dataset,
          n_epochs=1000, batch_size=64, patience_limit=10)

    # 5) Evaluate on test set
    metrics = evaluate_testset(model, "best_ppg2bpnet.pth", test_dataset)
