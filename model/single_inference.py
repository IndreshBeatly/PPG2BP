import torch
import os
import numpy as np
from ppg2bpnet import PPG2BPNetOptimized


def ensure_correct_ppg_shape(ppg_array: np.ndarray) -> np.ndarray:
    """
    Ensures the input PPG array is of shape (500,). Handles (1, 500), (500, 1), (1, 1, 500), etc.
    """
    ppg_array = np.squeeze(ppg_array)
    if ppg_array.shape != (500,):
        raise ValueError(f"Invalid PPG shape after squeeze: got {ppg_array.shape}, expected (500,)")
    return ppg_array


def single_inference(ppg_target_np: np.ndarray,
                     ppg_calib_np: np.ndarray,
                     sbp_calib_val: float,
                     dbp_calib_val: float,
                     checkpoint_path: str = "utils/best_ppg2bp_net.pth",
                     device: str = None) -> tuple[float, float]:
    """
    Run a single‐sample inference through PPG2BPNetOptimized.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = PPG2BPNetOptimized().to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Ensure input shapes are correct
    ppg_target_np = ensure_correct_ppg_shape(ppg_target_np)
    ppg_calib_np = ensure_correct_ppg_shape(ppg_calib_np)

    # Convert to tensors
    ppg_tgt_tensor = torch.from_numpy(ppg_target_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    ppg_cal_tensor = torch.from_numpy(ppg_calib_np.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    sbp_cal_tensor = torch.tensor([[sbp_calib_val]], dtype=torch.float32).to(device)
    dbp_cal_tensor = torch.tensor([[dbp_calib_val]], dtype=torch.float32).to(device)

    with torch.no_grad():
        out = model(ppg_tgt_tensor, ppg_cal_tensor, sbp_cal_tensor, dbp_cal_tensor)
        sbp_pred = out[0, 0].item()
        dbp_pred = out[0, 1].item()

    return sbp_pred, dbp_pred


if __name__ == "__main__":
    test_root = "data"
    case_id = "metadata"
    npz_path = os.path.join(test_root, case_id, "combined_ppg_bp4.npz")

    data = np.load(npz_path)
    ppg_windows = data["ppg_windows"]
    sbp_values = data["sbp_values"]
    dbp_values = data["dbp_values"]

    calib_index = 0
    ppg_calib_np = ppg_windows[calib_index]
    sbp_calib_val = float(sbp_values[calib_index])
    dbp_calib_val = float(dbp_values[calib_index])

    target_index = 1
    ppg_target_np = ppg_windows[target_index]
    target_sbp = float(sbp_values[target_index])
    target_dbp = float(dbp_values[target_index])

    sbp_out, dbp_out = single_inference(
        ppg_target_np,
        ppg_calib_np,
        sbp_calib_val,
        dbp_calib_val,
        checkpoint_path="utils/best_ppg2bp_net.pth"
    )

    print(f"Prediction for case '{case_id}', target_idx={target_index}:")
    print(f"  True SBP = {target_sbp:.2f} mmHg, Predicted SBP = {sbp_out:.2f} mmHg")
    print(f"  True DBP = {target_dbp:.2f} mmHg, Predicted DBP = {dbp_out:.2f} mmHg")
