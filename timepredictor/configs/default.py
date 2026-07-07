from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREDICT_ROOT = Path(__file__).resolve().parents[1]


class Config:
    project_root = PROJECT_ROOT
    predict_root = PREDICT_ROOT
    data_path = project_root / "datasets" / "FeatureExt.csv"
    output_root = predict_root / "Results"
    checkpoint_dir = predict_root / "weights"
    checkpoint_path = checkpoint_dir / "bilstm_cbam_bra_best.pth"
    yolo_weights = project_root / "weights" / "best.pt"
    infer_source = project_root / "inferimage"
    infer_output_dir = predict_root / "inference_results"
    feature_cols = [f"feature_{i + 1}" for i in range(128)]
    target_col = "remaining_time"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    test_ratio = 0.10
    train_ratio_within_trainval = 0.80
    predictable_classes = {"EM", "HM", "DT"}
    qualified_class = "ASM"
    overtime_class = "OD"
    layer_index = 21
    conf = 0.25
    iou_track_thresh = 0.30
    max_track_age = 2

    colors = {
        "red": "#E64B35",
        "blue": "#4DBBD5",
        "green": "#00A087",
        "dark_blue": "#3C5488",
        "orange": "#F39B7F",
        "purple": "#8491B4",
        "cyan": "#91D1C2",
    }

    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    batch_size = 32
    epochs = 200
    lr = 0.001
    early_stopping_patience = 20
    early_stopping_min_delta = 1e-4


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
