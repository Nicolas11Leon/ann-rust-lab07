//! Evaluation metrics for binary classification (Confusion Matrix, Accuracy, Precision, Recall, F1).

/// Binary confusion matrix (`Actual` vs `Predicted`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfusionMatrix {
    pub tn: usize,
    pub fp: usize,
    pub fn_count: usize,
    pub tp: usize,
}

impl ConfusionMatrix {
    /// Builds a confusion matrix from ground-truth labels (`0.0`/`1.0`) and predicted probabilities.
    pub fn from_predictions(actual: &[f64], predicted_probs: &[f64], threshold: f64) -> Self {
        assert_eq!(actual.len(), predicted_probs.len(), "Length mismatch");
        let mut tn = 0;
        let mut fp = 0;
        let mut fn_count = 0;
        let mut tp = 0;

        for (&y, &p) in actual.iter().zip(predicted_probs.iter()) {
            let actual_pos = y >= 0.5;
            let pred_pos = p >= threshold;
            match (actual_pos, pred_pos) {
                (false, false) => tn += 1,
                (false, true) => fp += 1,
                (true, false) => fn_count += 1,
                (true, true) => tp += 1,
            }
        }

        Self { tn, fp, fn_count, tp }
    }

    pub fn total(&self) -> usize {
        self.tn + self.fp + self.fn_count + self.tp
    }

    pub fn accuracy(&self) -> f64 {
        let tot = self.total();
        if tot == 0 {
            0.0
        } else {
            (self.tp + self.tn) as f64 / tot as f64
        }
    }

    pub fn precision(&self) -> f64 {
        let denom = self.tp + self.fp;
        if denom == 0 {
            0.0
        } else {
            self.tp as f64 / denom as f64
        }
    }

    pub fn recall(&self) -> f64 {
        let denom = self.tp + self.fn_count;
        if denom == 0 {
            0.0
        } else {
            self.tp as f64 / denom as f64
        }
    }

    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if (p + r) == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Prints the confusion matrix and validation metrics in a readable table.
    pub fn print_report(&self, title: &str) {
        println!("=== {} ===", title);
        println!("           Predicted");
        println!("Actual     0     1");
        println!("     0 {:>5} {:>5}", self.tn, self.fp);
        println!("     1 {:>5} {:>5}", self.fn_count, self.tp);
        println!("Accuracy : {:.4}", self.accuracy());
        println!("Precision: {:.4}", self.precision());
        println!("Recall   : {:.4}", self.recall());
        println!("F1-Score : {:.4}", self.f1_score());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_match_expected_values() {
        let actual = vec![0.0, 0.0, 1.0, 1.0];
        let probs = vec![0.1, 0.6, 0.4, 0.9];
        let cm = ConfusionMatrix::from_predictions(&actual, &probs, 0.5);

        assert_eq!(cm.tn, 1);
        assert_eq!(cm.fp, 1);
        assert_eq!(cm.fn_count, 1);
        assert_eq!(cm.tp, 1);
        assert!((cm.accuracy() - 0.5).abs() < 1e-12);
        assert!((cm.precision() - 0.5).abs() < 1e-12);
        assert!((cm.recall() - 0.5).abs() < 1e-12);
        assert!((cm.f1_score() - 0.5).abs() < 1e-12);
    }
}
