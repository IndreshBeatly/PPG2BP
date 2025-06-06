import torch
from torch.utils.data import DataLoader
from data_loader import PPG2BPDataset
from ppg2bpnet import PPG2BPNetOptimized
import numpy as np

def run_batch_inference(split_txt: str,
                        data_root: str = "data/final",
                        checkpoint_path: str = "best_ppg2bp_net.pth",
                        batch_size: int = 64,
                        num_workers: int = 2,
                        device: str = None):
    """
    Performs batch-by-batch inference on a PPG2BPDataset split.
    - split_txt:    Path to the .txt file listing subject IDs (e.g. split_test.txt).
    - data_root:    Root folder where “<subject_id>/signals_with_metadata.npz” live.
    - checkpoint_path: Path to the saved model weights (best_ppg2bp_net.pth).
    - batch_size:   Batch size for inference DataLoader.
    - num_workers:  Number of DataLoader workers.
    - device:       "cuda" or "cpu"; if None, inferred automatically.
    """

    # 1) Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 2) Load model and weights
    model = PPG2BPNetOptimized().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) Create Dataset & DataLoader for the requested split
    dataset = PPG2BPDataset(split_txt=split_txt, data_root=data_root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    all_sbp_preds = []
    all_dbp_preds = []
    all_sbp_trues = []   # optional: if you want to compare to ground truth
    all_dbp_trues = []   # optional

    # 4) Iterate batch-by-batch
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Move inputs to device
            ppg_tgt  = batch["ppg_target"].to(device)   # (B, 1, 500)
            ppg_cal  = batch["ppg_calib"].to(device)    # (B, 1, 500)
            sbp_cal  = batch["sbp_calib"].to(device)    # (B, 1)
            dbp_cal  = batch["dbp_calib"].to(device)    # (B, 1)

            # Forward pass → (B, 2)
            preds = model(ppg_tgt, ppg_cal, sbp_cal, dbp_cal)
            sbp_pred = preds[:, 0].cpu().numpy()   # shape: (B,)
            dbp_pred = preds[:, 1].cpu().numpy()   # shape: (B,)

            all_sbp_preds.append(sbp_pred)
            all_dbp_preds.append(dbp_pred)

            # If you want to collect ground-truths:
            sbp_true = batch["sbp_true"].view(-1).cpu().numpy()
            dbp_true = batch["dbp_true"].view(-1).cpu().numpy()
            all_sbp_trues.append(sbp_true)
            all_dbp_trues.append(dbp_true)

            # (Optional) Print a few predictions for this batch:
            print(f"Batch {batch_idx} → SBP_preds: {sbp_pred[:5].tolist()}  DBP_preds: {dbp_pred[:5].tolist()}")

    # 5) Concatenate all batches
    all_sbp_preds = np.concatenate(all_sbp_preds, axis=0)   # (num_samples,)
    all_dbp_preds = np.concatenate(all_dbp_preds, axis=0)
    all_sbp_trues = np.concatenate(all_sbp_trues, axis=0)
    all_dbp_trues = np.concatenate(all_dbp_trues, axis=0)

    # 6) (Optional) Compute overall MAE on this split
    mae_sbp = np.mean(np.abs(all_sbp_preds - all_sbp_trues))
    mae_dbp = np.mean(np.abs(all_dbp_preds - all_dbp_trues))
    print(f"\n→ Inference complete on {split_txt}")
    print(f"   Total samples: {len(all_sbp_preds)}")
    print(f"   MAE (SBP): {mae_sbp:.2f} mmHg   MAE (DBP): {mae_dbp:.2f} mmHg")

    return all_sbp_preds, all_dbp_preds


if __name__ == "__main__":
    # Example usage: batch inference on the test split
    sbp_predictions, dbp_predictions = run_batch_inference(
        split_txt="data/data_split/split_test.txt",
        data_root="data/final",
        checkpoint_path="best_ppg2bp_net.pth",
        batch_size=64,
        num_workers=2
    )

    # (Optional) Save the predictions to disk, e.g. as a .npz
    np.savez(
        "test_split_predictions.npz",
        sbp_pred=sbp_predictions,
        dbp_pred=dbp_predictions
    )
    print("\nSaved SBP/DBP predictions to “test_split_predictions.npz”")
