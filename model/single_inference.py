import torch
import numpy as np
from ppg2bpnet import PPG2BPNetOptimized

def single_inference(ppg_target_np: np.ndarray,
                     ppg_calib_np: np.ndarray,
                     sbp_calib_val: float,
                     dbp_calib_val: float,
                     checkpoint_path: str = "best_ppg2bp_net.pth",
                     device: str = None) -> tuple[(float, float)]:
    """
    Run a single‐sample inference through PPG2BPNetOptimized.

    Args:
      ppg_target_np:   NumPy array of shape (500,) for the target PPG window.
      ppg_calib_np:    NumPy array of shape (500,) for the calibration PPG window.
      sbp_calib_val:   Calibration SBP value (scalar, e.g. 120.0).
      dbp_calib_val:   Calibration DBP value (scalar, e.g. 80.0).
      checkpoint_path: Path to "best_ppg2bp_net.pth".
      device:          "cuda" or "cpu". If None, auto‐selects.

    Returns:
      (sbp_pred, dbp_pred): Tuple of floats, the predicted SBP/DBP in mmHg.
    """

    # 1) Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # 2) Load model & weights
    model = PPG2BPNetOptimized().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3) Preprocess / convert to torch.Tensor
    #    Assumes ppg_target_np and ppg_calib_np are already float32 and normalized as in training.
    #    If not, insert the same preprocessing (e.g. (x - mean) / std) here.
    ppg_tgt_tensor = torch.from_numpy(ppg_target_np.astype(np.float32)) \
                          .unsqueeze(0).unsqueeze(0).to(device)
    # → shape: (1, 1, 500)
    ppg_cal_tensor = torch.from_numpy(ppg_calib_np.astype(np.float32)) \
                          .unsqueeze(0).unsqueeze(0).to(device)
    # → shape: (1, 1, 500)
    sbp_cal_tensor = torch.tensor([[sbp_calib_val]], dtype=torch.float32).to(device)
    # → shape: (1, 1)
    dbp_cal_tensor = torch.tensor([[dbp_calib_val]], dtype=torch.float32).to(device)
    # → shape: (1, 1)

    # 4) Forward pass under no_grad
    with torch.no_grad():
        out = model(ppg_tgt_tensor, ppg_cal_tensor, sbp_cal_tensor, dbp_cal_tensor)
        sbp_pred = out[0, 0].item()
        dbp_pred = out[0, 1].item()

    return sbp_pred, dbp_pred


if __name__ == "__main__":
    # ───── Example usage ─────
    # Replace these with your actual PPG data and calibration values:
    ppg_target_np = np.random.randn(500).astype(np.float32)  # placeholder
    ppg_calib_np  = np.random.randn(500).astype(np.float32)  # placeholder
    sbp_calib_val = 120.0
    dbp_calib_val = 80.0

    sbp_out, dbp_out = single_inference(
        ppg_target_np,
        ppg_calib_np,
        sbp_calib_val,
        dbp_calib_val,
        checkpoint_path="best_ppg2bp_net.pth"
    )

    print("Single‐Sample Inference Result:")
    print(f"  SBP_predicted = {sbp_out:.2f} mmHg")
    print(f"  DBP_predicted = {dbp_out:.2f} mmHg")
