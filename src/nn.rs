//! Feed-Forward Artificial Neural Network (MLP) with Sigmoid activation and Backpropagation.

use crate::preprocessing::Xorshift64;

/// A single dense layer storing `weights[neuron][input]` and `biases[neuron]`.
#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

/// Multi-layer feed-forward neural network for binary classification.
#[derive(Debug, Clone)]
pub struct NeuralNet {
    pub layer_sizes: Vec<usize>,
    pub layers: Vec<Layer>,
    pub learning_rate: f64,
    pub final_error: f64,
    pub steps: usize,
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn sigmoid_derivative_from_output(y: f64) -> f64 {
    y * (1.0 - y)
}

impl NeuralNet {
    /// Creates a new feed-forward network with `layer_sizes` (e.g. `&[4, 3, 1]`).
    pub fn new(layer_sizes: &[usize], learning_rate: f64, seed: u64) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Network must have at least an input and an output layer"
        );
        let mut rng = Xorshift64::new(seed);
        let mut layers = Vec::with_capacity(layer_sizes.len() - 1);

        for pair in layer_sizes.windows(2) {
            let in_dim = pair[0];
            let out_dim = pair[1];
            let limit = (6.0 / (in_dim + out_dim) as f64).sqrt();

            let mut weights = Vec::with_capacity(out_dim);
            let mut biases = Vec::with_capacity(out_dim);
            for _ in 0..out_dim {
                let row: Vec<f64> = (0..in_dim)
                    .map(|_| rng.gen_range_f64(-limit, limit))
                    .collect();
                weights.push(row);
                biases.push(rng.gen_range_f64(-limit, limit));
            }
            layers.push(Layer { weights, biases });
        }

        Self {
            layer_sizes: layer_sizes.to_vec(),
            layers,
            learning_rate,
            final_error: f64::INFINITY,
            steps: 0,
        }
    }

    /// Forward pass returning activations for every layer (including the input at index 0).
    fn forward_all(&self, input: &[f64]) -> Vec<Vec<f64>> {
        assert_eq!(input.len(), self.layer_sizes[0], "Input dimension mismatch");
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        activations.push(input.to_vec());

        for layer in &self.layers {
            let prev = activations.last().unwrap();
            let mut next = Vec::with_capacity(layer.weights.len());
            for (w_row, &b) in layer.weights.iter().zip(layer.biases.iter()) {
                let net_input: f64 = w_row.iter().zip(prev.iter()).map(|(&w, &x)| w * x).sum::<f64>() + b;
                next.push(sigmoid(net_input));
            }
            activations.push(next);
        }

        activations
    }

    /// Computes the network output probability for a single input tuple.
    pub fn compute(&self, input: &[f64]) -> f64 {
        let activations = self.forward_all(input);
        activations.last().unwrap()[0]
    }

    /// Computes output probabilities for a batch of input tuples.
    pub fn predict_batch(&self, inputs: &[Vec<f64>]) -> Vec<f64> {
        inputs.iter().map(|row| self.compute(row)).collect()
    }

    /// Computes input feature importance as the sum of absolute outgoing weights in layer 0.
    pub fn input_feature_importance(&self) -> Vec<f64> {
        let in_dim = self.layer_sizes[0];
        let mut importance = vec![0.0; in_dim];
        for w_row in &self.layers[0].weights {
            for (i, &w) in w_row.iter().enumerate() {
                importance[i] += w.abs();
            }
        }
        importance
    }

    /// Trains the network using backpropagation until `max_epochs` or `error_threshold` is reached.
    ///
    /// Tracks `0.5 * sum((target - output)^2)` as `final_error` (matching R's `neuralnet` SSE/2).
    pub fn fit(
        &mut self,
        inputs: &[Vec<f64>],
        targets: &[f64],
        max_epochs: usize,
        error_threshold: f64,
    ) {
        assert_eq!(inputs.len(), targets.len(), "Inputs and targets length mismatch");
        assert!(!inputs.is_empty(), "Cannot train on empty dataset");

        let num_layers = self.layers.len();

        for epoch in 1..=max_epochs {
            let mut total_sse = 0.0;

            for (x, &y_true) in inputs.iter().zip(targets.iter()) {
                let activations = self.forward_all(x);
                let y_pred = activations.last().unwrap()[0];
                let diff = y_true - y_pred;
                total_sse += 0.5 * diff * diff;

                // Compute deltas backwards from output layer to first hidden layer
                let mut deltas: Vec<Vec<f64>> = vec![Vec::new(); num_layers];
                let out_delta = diff * sigmoid_derivative_from_output(y_pred);
                deltas[num_layers - 1] = vec![out_delta];

                for l in (0..num_layers - 1).rev() {
                    let curr_out = &activations[l + 1];
                    let next_layer = &self.layers[l + 1];
                    let next_deltas = &deltas[l + 1];

                    let mut layer_deltas = Vec::with_capacity(curr_out.len());
                    for (i, &act) in curr_out.iter().enumerate() {
                        let back_err: f64 = next_layer
                            .weights
                            .iter()
                            .zip(next_deltas.iter())
                            .map(|(w_row, &d)| w_row[i] * d)
                            .sum();
                        layer_deltas.push(back_err * sigmoid_derivative_from_output(act));
                    }
                    deltas[l] = layer_deltas;
                }

                // Update weights and biases
                for l in 0..num_layers {
                    let prev_act = &activations[l];
                    for (j, &delta) in deltas[l].iter().enumerate() {
                        for (i, &a) in prev_act.iter().enumerate() {
                            self.layers[l].weights[j][i] += self.learning_rate * delta * a;
                        }
                        self.layers[l].biases[j] += self.learning_rate * delta;
                    }
                }
            }

            self.final_error = total_sse;
            self.steps = epoch;

            if total_sse <= error_threshold {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn network_learns_xor_problem() {
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![0.0, 1.0, 1.0, 0.0];

        let mut nn = NeuralNet::new(&[2, 4, 1], 0.5, 42);
        nn.fit(&inputs, &targets, 10_000, 0.01);

        assert!(nn.compute(&[0.0, 0.0]) < 0.2);
        assert!(nn.compute(&[0.0, 1.0]) > 0.8);
        assert!(nn.compute(&[1.0, 0.0]) > 0.8);
        assert!(nn.compute(&[1.0, 1.0]) < 0.2);
    }
}
