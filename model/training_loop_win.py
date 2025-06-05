import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your Dataset and Model classes (make sure these imports work relative to your file structure)
from data_loader import PPG2BPDataset
from ppg2bpnet import PPG2BPNetOptimized


def combined_loss(preds, truths, device):
    """
    Computes: (MSE_SBP + MSE_DBP) + 0.5*(MAE_SBP + MAE_DBP)
    preds: (batch, 2) ⇒ [sbp_pred, dbp_pred]
    truths: dict with keys "sbp_true", "dbp_true", each shape (batch, 1)
    """
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    
    sbp_pred = preds[:, 0]
    dbp_pred = preds[:, 1]
    sbp_true = truths["sbp_true"].view(-1).to(device)
    dbp_true = truths["dbp_true"].view(-1).to(device)

    loss_sbp_mse = mse_loss(sbp_pred, sbp_true)
    loss_dbp_mse = mse_loss(dbp_pred, dbp_true)
    loss_sbp_mae = mae_loss(sbp_pred, sbp_true)
    loss_dbp_mae = mae_loss(dbp_pred, dbp_true)

    return (loss_sbp_mse + loss_dbp_mse) + 0.5 * (loss_sbp_mae + loss_dbp_mae)


def main():
    # 1) Instantiate Datasets + DataLoaders
    train_dataset = PPG2BPDataset(
        split_txt="data/data_split/split_train.txt",
        data_root="data/final"
    )
    val_dataset = PPG2BPDataset(
        split_txt="data/data_split/split_val.txt",
        data_root="data/final"
    )
    test_dataset = PPG2BPDataset(
        split_txt="data/data_split/split_test.txt",
        data_root="data/final"
    )

    # On Windows, set num_workers=0 or wrap DataLoader in __main__
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,   # Must be ≥0, but script is inside __main__ so it will work
        pin_memory=False # pin_memory only helps if you have a CUDA device; safe to set False on CPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2,
        pin_memory=False
    )

    # 2) Instantiate Model, Optimizer, Scheduler
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PPG2BPNetOptimized().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )

    # 3) Early Stopping Setup
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    max_epochs = 100

    # 4) Training Loop
    for epoch in range(max_epochs):
        if early_stop:
            print(f"Early stopping triggered after epoch {epoch} (no improvement for {patience} epochs).")
            break

        # ---- Training Phase ----
        model.train()
        running_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Train]"):
            optimizer.zero_grad()
            ppg_tgt = batch["ppg_target"].to(device)   # (B, 1, 500)
            ppg_cal = batch["ppg_calib"].to(device)    # (B, 1, 500)
            sbp_cal = batch["sbp_calib"].to(device)    # (B, 1)
            dbp_cal = batch["dbp_calib"].to(device)    # (B, 1)
            truths = {
                "sbp_true": batch["sbp_true"].to(device),
                "dbp_true": batch["dbp_true"].to(device),
            }

            preds = model(ppg_tgt, ppg_cal, sbp_cal, dbp_cal)  # → (B, 2)
            loss = combined_loss(preds, truths, device)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * ppg_tgt.size(0)

        epoch_train_loss = running_train_loss / len(train_loader.dataset)

        # ---- Validation Phase ----
        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{max_epochs} [Val]"):
                ppg_tgt = batch["ppg_target"].to(device)
                ppg_cal = batch["ppg_calib"].to(device)
                sbp_cal = batch["sbp_calib"].to(device)
                dbp_cal = batch["dbp_calib"].to(device)
                truths = {
                    "sbp_true": batch["sbp_true"].to(device),
                    "dbp_true": batch["dbp_true"].to(device),
                }

                preds = model(ppg_tgt, ppg_cal, sbp_cal, dbp_cal)
                loss = combined_loss(preds, truths, device)
                running_val_loss += loss.item() * ppg_tgt.size(0)

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        print(f"[Epoch {epoch+1}] Train Loss: {epoch_train_loss:.4f}   Val Loss: {epoch_val_loss:.4f}")

        # Step the scheduler on validation loss
        scheduler.step(epoch_val_loss)

        # Early stopping logic
        if epoch_val_loss < best_val_loss - 1e-6:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_ppg2bp_net.pth")
            print(f" → New best Val Loss: {best_val_loss:.4f} (checkpoint saved)")
        else:
            epochs_no_improve += 1
            print(f" → No improvement for {epochs_no_improve}/{patience} epochs")

        if epochs_no_improve >= patience:
            early_stop = True

    # ---- Testing Phase ----
    print("\n--- Testing on split_test ---")
    model.load_state_dict(torch.load("best_ppg2bp_net.pth"))
    model.eval()

    test_mse_sbp = 0.0
    test_mse_dbp = 0.0
    test_mae_sbp = 0.0
    test_mae_dbp = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test"):
            ppg_tgt = batch["ppg_target"].to(device)
            ppg_cal = batch["ppg_calib"].to(device)
            sbp_cal = batch["sbp_calib"].to(device)
            dbp_cal = batch["dbp_calib"].to(device)
            sbp_true = batch["sbp_true"].view(-1).to(device)
            dbp_true = batch["dbp_true"].view(-1).to(device)

            preds = model(ppg_tgt, ppg_cal, sbp_cal, dbp_cal)  # (B,2)
            sbp_pred = preds[:, 0]
            dbp_pred = preds[:, 1]

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

    print(f"\nTest SBP  → MSE: {test_mse_sbp:.4f}  MAE: {test_mae_sbp:.4f}")
    print(f"Test DBP  → MSE: {test_mse_dbp:.4f}  MAE: {test_mae_dbp:.4f}")


if __name__ == "__main__":
    main()
