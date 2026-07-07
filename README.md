# CDS-YOLO-LSTM

A lightweight vision-based deep learning framework for real-time moisture estimation and drying time prediction in maize.

The framework consists of two sequential stages:

- **Stage I:** CDS-YOLO for maize kernel detection and moisture-state recognition.
- **Stage II:** CDS-YOLO-LSTM for remaining drying-time prediction.

---

# Repository Structure

```text
CDS-YOLO-LSTM
│
├── ultralytics/                 # Stage I: CDS-YOLO
│   ├── cfg/                     # Model configurations
│   ├── nn/                      # Core network implementations
│   ├── ...
│
├── timepredictor/              # Stage II: CDS-YOLO-LSTM
│   ├── configs/                # Hyperparameter settings
│   ├── data/                   # Dataset loading and preprocessing
│   ├── engine/                 # Training, validation, testing,
│   │                             result visualization,
│   │                             logging and report generation
│   ├── models/                 # CNN-BiLSTM and CBAM-BRA modules
│   ├── ...
│
├── datasets/
│   └── dataset/                # The dataset is currently not publicly available
│
└── README.md
```

---

# Environment

Recommended environment:

```text
Python >= 3.8
Torch >= 2.0.0
CUDA >= 11.8
numpy >= 1.24.2
torchvision >= 0.15.1
```

---

# Training


```bash
python ultralytics/stage1train.py   #stage1:CDS-YOLO
```

```bash
python timepredictor/stage2train.py   #stage2:CDS-YOLO-LSTM
```

---

# Dataset Availability

The maize drying dataset used in this study is currently not publicly available, as it is being utilized in ongoing projects within our research group.

Researchers interested in collaboration may contact the corresponding author.

---

## Acknowledgements

The Stage-I implementation is developed based on the official Ultralytics/YOLO framework released under the AGPL-3.0 license.

Additional modules including SHSA, EMCA, CACS and UIoU were incorporated and adapted for maize moisture-state estimation.

The Stage-II prediction module was independently implemented for drying-time prediction.