# Model Comparison Report — Advertisement Dataset (Lab 04 & Lab 07)

Dataset: `data/advertisement.csv` (399 samples, features: `Age`, `EstimatedSalary`, target: `Purchased`, 75/25 train-test split with `seed = 42`, Min-Max `[0, 1]` scaling fitted on the training split).

## Validation Metrics (Test Set, n = 100)

| Model | Ecosystem | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **ANN (2 → 4 → 1 MLP)** | Pure Rust (`src/nn.rs`) | **0.8700** | **0.7027** | **0.9286** | **0.8000** |
| **SVM / SVC (RBF Kernel)** | Rust `smartcore` 0.7 | 0.8600 | 0.6944 | 0.8929 | 0.7812 |
| **Decision Tree** | Rust `smartcore` 0.7 | 0.8000 | 0.6250 | 0.7143 | 0.6667 |
| *Decision Tree (Lab 04 Ref)* | *R `rpart` (Figure 3)* | *0.7800* | *0.6944* | *0.6944* | *0.6944* |

## Confusion Matrices (Test Set)

- **ANN (Pure Rust MLP):** `TN = 61`, `FP = 11`, `FN = 2`, `TP = 26`
- **SVM / SVC (SmartCore):** `TN = 61`, `FP = 11`, `FN = 3`, `TP = 25`
- **Decision Tree (SmartCore):** `TN = 60`, `FP = 12`, `FN = 8`, `TP = 20`

## Key Takeaways

1. Both non-linear models (**ANN** and **RBF SVM**) substantially outperform the axis-aligned **Decision Tree** on `Age` vs `EstimatedSalary`, where the purchase boundary is curved.
2. The **Pure Rust ANN** achieves the highest **Accuracy (87.0%)**, **Recall (92.9%)**, and **F1-Score (0.8000)**, minimizing false negatives (`FN = 2`).
