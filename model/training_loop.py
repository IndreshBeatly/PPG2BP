import torch
from torch.utils.data import DataLoader
from data_loader import PPG2BPDataset
from ppg2bpnet import PPG2BPNetOptimized
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F

# -------------------
# 1) Instantiate Datasets + DataLoaders
# -------------------
train_dataset = PPG2BPDataset(split_txt="data/data_split/split_train.txt", data_root="data/final")
val_dataset   = PPG2BPDataset(split_txt="data/data_split/split_val.txt",   data_root="data/final")
test_dataset  = PPG2BPDataset(split_txt="data/data_split/split_test.txt",  data_root="data/final")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# -------------------
# 2) Instantiate Model, Optimizer, Loss, Scheduler
# -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PPG2BPNetOptimized().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                 factor=0.5, patience=5)

mse_loss = nn.MSELoss()
mae_loss = nn.L1Loss()
alpha = 0.5

def combined_loss(preds, truths):
    sbp_pred = preds[:, 0]
    dbp_pred = preds[:, 1]
    sbp_true = truths["sbp_true"].view(-1).to(device)
    dbp_true = truths["dbp_true"].view(-1).to(device)

    loss_sbp_mse = mse_loss(sbp_pred, sbp_true)
    loss_dbp_mse = mse_loss(dbp_pred, dbp_true)
    loss_sbp_mae = mae_loss(sbp_pred, sbp_true)
    loss_dbp_mae = mae_loss(dbp_pred, dbp_true)

    return (loss_sbp_mse + loss_dbp_mse) + alpha * (loss_sbp_mae + loss_dbp_mae)


# 3) Early Stopping Parameters
patience = 10           # how many epochs to wait after last improvement
best_val_loss = float('inf')
epochs_no_improve = 0   # counter for epochs since last improvement
early_stop = False
max_epochs = 100

for epoch in range(max_epochs):
    if early_stop:
        print(f"Stopping early after epoch {epoch} (no improvement for {patience} epochs).")
        break

    # ---------- Training Phase ----------
    model.train()
    running_train_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]"):
        optimizer.zero_grad()
        ppg_tgt  = batch["ppg_target"].to(device)   # (batch,1,500)
        ppg_cal  = batch["ppg_calib"].to(device)    # (batch,1,500)
        sbp_cal  = batch["sbp_calib"].to(device)    # (batch,1)
        dbp_cal  = batch["dbp_calib"].to(device)    # (batch,1)
        truths   = {
            "sbp_true": batch["sbp_true"].to(device),
            "dbp_true": batch["dbp_true"].to(device),
        }

        preds = model(ppg_tgt, ppg_cal, sbp_cal, dbp_cal)  # → (batch, 2)
        loss = combined_loss(preds, truths)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item() * ppg_tgt.size(0)

    epoch_train_loss = running_train_loss / len(train_loader.dataset)

    # ---------- Validation Phase ----------
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
            ppg_tgt  = batch["ppg_target"].to(device)
            ppg_cal  = batch["ppg_calib"].to(device)
            sbp_cal  = batch["sbp_calib"].to(device)
            dbp_cal  = batch["dbp_calib"].to(device)
            truths   = {
                "sbp_true": batch["sbp_true"].to(device),
                "dbp_true": batch["dbp_true"].to(device),
            }

            preds = model(ppg_tgt, ppg_cal, sbp_cal, dbp_cal)
            loss  = combined_loss(preds, truths)
            running_val_loss += loss.item() * ppg_tgt.size(0)

    epoch_val_loss = running_val_loss / len(val_loader.dataset)

    print(f"[Epoch {epoch+1}] Train Loss: {epoch_train_loss:.4f}   Val Loss: {epoch_val_loss:.4f}")

    # Step the scheduler based on validation loss
    scheduler.step(epoch_val_loss)

    # Check for improvement
    if epoch_val_loss < best_val_loss - 1e-6:
        # Improvement found → save checkpoint, reset counter
        best_val_loss = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), "best_ppg2bp_net.pth")
        print(f" → New best validation loss: {best_val_loss:.4f} (checkpoint saved)")
    else:
        # No improvement this epoch
        epochs_no_improve += 1
        print(f" → No improvement for {epochs_no_improve}/{patience} epochs")

    # If we’ve gone `patience` epochs with no improvement, stop early
    if epochs_no_improve >= patience:
        early_stop = True

# 4) After Early Stopping, Evaluate on Test Set
print("\n=== Testing on split_test ===")
# Load best model checkpoint
model.load_state_dict(torch.load("best_ppg2bp_net.pth"))
model.eval()

test_mse_sbp = 0.0
test_mse_dbp = 0.0
test_mae_sbp = 0.0
test_mae_dbp = 0.0
total_samples = 0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        ppg_tgt  = batch["ppg_target"].to(device)
        ppg_cal  = batch["ppg_calib"].to(device)
        sbp_cal  = batch["sbp_calib"].to(device)
        dbp_cal  = batch["dbp_calib"].to(device)
        sbp_true = batch["sbp_true"].view(-1).to(device)
        dbp_true = batch["dbp_true"].view(-1).to(device)

        preds = model(ppg_tgt, ppg_cal, sbp_cal, dbp_cal)  # (B,2)
        sbp_pred = preds[:,0]
        dbp_pred = preds[:,1]

        # MSE
        test_mse_sbp += F.mse_loss(sbp_pred, sbp_true, reduction="sum").item()
        test_mse_dbp += F.mse_loss(dbp_pred, dbp_true, reduction="sum").item()
        # MAE
        test_mae_sbp += F.l1_loss(sbp_pred, sbp_true, reduction="sum").item()
        test_mae_dbp += F.l1_loss(dbp_pred, dbp_true, reduction="sum").item()

        total_samples += ppg_tgt.size(0)

test_mse_sbp /= total_samples
test_mse_dbp /= total_samples
test_mae_sbp /= total_samples
test_mae_dbp /= total_samples

print(f"\nTest SBP  → MSE: {test_mse_sbp:.4f},  MAE: {test_mae_sbp:.4f}")
print(f"Test DBP  → MSE: {test_mse_dbp:.4f},  MAE: {test_mae_dbp:.4f}")