# ANN in Rust — Lab 07

A **zero-dependency, pure-Rust** implementation of a Multi-Layer Perceptron (Feed-Forward Artificial Neural Network) trained with Backpropagation, built as a portfolio project replicating **Lab 07: A "gentle" Introduction to Artificial Neural Networks**.

## Features

- **Pure Standard Library (`std`)**: No external ML or linear algebra crates required.
- **Data Preprocessing (`preprocessing.rs`)**: CSV loader, Min-Max normalization to `[0, 1]` with reusable scaling parameters (`mins`, `maxs`) for inference on new tuples, and deterministic 70/30 train/test splitting (`seed = 42`).
- **Neural Network Engine (`nn.rs`)**: Configurable multi-layer feed-forward network, sigmoid activation, backpropagation training, error tracking, and inference (`compute`).
- **Evaluation Metrics (`metrics.rs`)**: Confusion matrix, Accuracy, Precision, Recall, and F1-Score.
- **Visualization (`viz.rs`)**: Exports an SVG diagram of the trained network architecture (nodes, connection weights, and bias terms), mirroring `plot(nn)` in R's `neuralnet`.

## Usage

```bash
# Run unit & integration tests
cargo test

# Run the Iris baseline experiment (Lab 07 Sections 2-7)
cargo run -- iris
```

## Project Roadmap (User Stories)

- **US-01**: Project setup, modular crate structure, and portfolio documentation
- **US-02**: CSV loading, Min-Max `[0, 1]` normalization, and deterministic 70/30 split
- **US-03**: Feed-Forward ANN core with Sigmoid activation and Backpropagation
- **US-04**: Iris (`is_setosa`) baseline replication, SVG architecture export, and new-tuple inference
- **US-05**: Autonomous Work 1 & 2 — Advertisement dataset & benchmark vs Decision Tree / SVM
- **US-06**: Autonomous Work 3 — Interpretable ANN classifier on a UCI ML Repository dataset
