import torch
import torch.nn as nn

class PPG2BPNetOptimized(nn.Module):
    """
    An optimized PyTorch implementation of PPG2BP-Net (Joung et al., 2023),
    tuned for 50 Hz × 10 s (500‐sample) PPG windows, with hyperparameters
    chosen for maximum performance on ~300 windows/subject data.
    """

    def __init__(self):
        super().__init__()

        # 1) Shared 1D‐CNN Branch (four Conv blocks → Global AvgPool → Dropout → FC → 16)
        self.cnn_branch = nn.Sequential(
            # ─── Block 1 ───
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),

            # ─── Block 2 ───
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=7, padding=3, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),

            # ─── Block 3 ───
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            # ─── Block 4 ───
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            # ─── Global Average Pooling in time (500 → 1) ───
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        # After this → shape: (batch, 256, 1)
        self.cnn_fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Flatten(start_dim=1),                # → (batch, 256)
            nn.Linear(in_features=256, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)                   # → (batch, 16)
        )

        # 2) Calibration MLP (for SBP_calib and DBP_calib separately)
        #    Each pipeline: 1 → 32 → 16 → features
        self.mlp_sbp = nn.Sequential(
            nn.Linear(in_features=1, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.mlp_dbp = nn.Sequential(
            nn.Linear(in_features=1, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=16),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )

        # 3) Final Fusion & Output (concatenate 16+16+16+16+16 = 80 dims → 64 → 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(in_features=80, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=2)
        )

    def forward(self,
                ppg_target: torch.Tensor,
                ppg_calib:  torch.Tensor,
                sbp_calib:  torch.Tensor,
                dbp_calib:  torch.Tensor) -> torch.Tensor:
        """
        Inputs:
          - ppg_target: Tensor (batch, 1, 500)
          - ppg_calib:  Tensor (batch, 1, 500)
          - sbp_calib:  Tensor (batch, 1) or (batch,)  
          - dbp_calib:  Tensor (batch, 1) or (batch,)
        Returns:
          - out: Tensor (batch, 2)  => (SBP_pred, DBP_pred)
        """
        batch_size = ppg_target.shape[0]

        ### 1. Pass through shared 1D-CNN branch
        x_t = self.cnn_branch(ppg_target)   # → (batch, 256, 1)
        x_t = self.cnn_fc(x_t)             # → (batch, 16)

        x_c = self.cnn_branch(ppg_calib)    # → (batch, 256, 1)
        x_c = self.cnn_fc(x_c)             # → (batch, 16)

        # 2. Absolute difference (batch, 16) 
        x_diff = torch.abs(x_t - x_c)       # → (batch, 16)

        # 3. Calibration MLP
        sbp_in = sbp_calib.view(batch_size, 1).float()  # (batch,1)
        dbp_in = dbp_calib.view(batch_size, 1).float()  # (batch,1)
        f_sbp = self.mlp_sbp(sbp_in)                    # → (batch, 16)
        f_dbp = self.mlp_dbp(dbp_in)                    # → (batch, 16)

        # 4. Concatenate [x_t (16), x_c (16), x_diff (16), f_sbp (16), f_dbp (16)] → (batch, 80)
        fused = torch.cat([x_t, x_c, x_diff, f_sbp, f_dbp], dim=1)  # → (batch, 80)

        # 5. Final fusion to (batch,2)
        out = self.fusion_layer(fused)  # → (batch, 2)
        return out
