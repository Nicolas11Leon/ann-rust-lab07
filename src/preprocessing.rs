//! Data loading, Min-Max normalization to [0, 1], and deterministic train/test splitting.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Tabular dataset with numeric features and binary targets (0.0 or 1.0).
#[derive(Debug, Clone)]
pub struct Dataset {
    pub feature_names: Vec<String>,
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<f64>,
}

/// Min-Max scaler mapping each feature column to `[0.0, 1.0]`.
#[derive(Debug, Clone)]
pub struct MinMaxScaler {
    pub mins: Vec<f64>,
    pub maxs: Vec<f64>,
}

impl MinMaxScaler {
    /// Computes per-column minimum and maximum values from `features`.
    pub fn fit(features: &[Vec<f64>]) -> Self {
        assert!(!features.is_empty(), "Cannot fit scaler on empty dataset");
        let n_cols = features[0].len();
        let mut mins = vec![f64::INFINITY; n_cols];
        let mut maxs = vec![f64::NEG_INFINITY; n_cols];

        for row in features {
            assert_eq!(row.len(), n_cols, "Inconsistent feature count in row");
            for (j, &val) in row.iter().enumerate() {
                if val < mins[j] {
                    mins[j] = val;
                }
                if val > maxs[j] {
                    maxs[j] = val;
                }
            }
        }

        Self { mins, maxs }
    }

    /// Scales a single observation using previously fitted `mins` and `maxs`.
    pub fn transform_row(&self, row: &[f64]) -> Vec<f64> {
        assert_eq!(row.len(), self.mins.len(), "Feature count mismatch");
        row.iter()
            .enumerate()
            .map(|(j, &val)| {
                let range = self.maxs[j] - self.mins[j];
                if range.abs() < f64::EPSILON {
                    0.0
                } else {
                    (val - self.mins[j]) / range
                }
            })
            .collect()
    }

    /// Scales all rows in `features` to `[0.0, 1.0]`.
    pub fn transform(&self, features: &[Vec<f64>]) -> Vec<Vec<f64>> {
        features.iter().map(|r| self.transform_row(r)).collect()
    }
}

/// Deterministic Xorshift64 PRNG for reproducible shuffling and weight initialization.
#[derive(Debug, Clone)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Creates a new PRNG with a non-zero seed.
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed },
        }
    }

    /// Generates the next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generates a uniform `f64` in `[0.0, 1.0)`.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    /// Generates a uniform `f64` in `[low, high)`.
    pub fn gen_range_f64(&mut self, low: f64, high: f64) -> f64 {
        low + self.next_f64() * (high - low)
    }
}

/// Loads a CSV file with a header row, extracting numeric feature columns and a binary target.
///
/// Any column in `feature_indices` is parsed as `f64`. The `target_index` column is mapped to
/// `1.0` if it matches `positive_label` (trimmed, quotes stripped), else `0.0`.
pub fn load_csv<P: AsRef<Path>>(
    path: P,
    feature_indices: &[usize],
    target_index: usize,
    positive_label: &str,
) -> Result<Dataset, String> {
    let file = File::open(path.as_ref())
        .map_err(|e| format!("Failed to open {:?}: {}", path.as_ref(), e))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let header_line = lines
        .next()
        .ok_or_else(|| "CSV file is empty".to_string())?
        .map_err(|e| format!("Failed to read header: {}", e))?;

    let headers: Vec<String> = header_line
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .collect();

    let feature_names: Vec<String> = feature_indices
        .iter()
        .map(|&idx| {
            headers
                .get(idx)
                .cloned()
                .ok_or_else(|| format!("Feature index {} out of bounds", idx))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let mut features = Vec::new();
    let mut targets = Vec::new();

    for (line_no, line_res) in lines.enumerate() {
        let line = line_res.map_err(|e| format!("Read error at line {}: {}", line_no + 2, e))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let cols: Vec<&str> = trimmed.split(',').collect();

        let mut row = Vec::with_capacity(feature_indices.len());
        for &idx in feature_indices {
            let raw = cols
                .get(idx)
                .ok_or_else(|| format!("Missing column {} at line {}", idx, line_no + 2))?
                .trim()
                .trim_matches('"');
            let val: f64 = raw.parse().map_err(|_| {
                format!(
                    "Invalid float '{}' at line {}, col {}",
                    raw,
                    line_no + 2,
                    idx
                )
            })?;
            row.push(val);
        }

        let target_raw = cols
            .get(target_index)
            .ok_or_else(|| format!("Missing target column {} at line {}", target_index, line_no + 2))?
            .trim()
            .trim_matches('"');
        let target = if target_raw == positive_label { 1.0 } else { 0.0 };

        features.push(row);
        targets.push(target);
    }

    Ok(Dataset {
        feature_names,
        features,
        targets,
    })
}

/// Splits `features` and `targets` into `(train_x, train_y, test_x, test_y)` deterministically.
pub fn train_test_split(
    features: &[Vec<f64>],
    targets: &[f64],
    train_ratio: f64,
    seed: u64,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>) {
    assert_eq!(features.len(), targets.len(), "Features and targets length mismatch");
    assert!(
        (0.0..1.0).contains(&train_ratio),
        "train_ratio must be in (0.0, 1.0)"
    );

    let n = features.len();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = Xorshift64::new(seed);

    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        indices.swap(i, j);
    }

    let train_len = (n as f64 * train_ratio).round() as usize;
    let mut train_x = Vec::with_capacity(train_len);
    let mut train_y = Vec::with_capacity(train_len);
    let mut test_x = Vec::with_capacity(n - train_len);
    let mut test_y = Vec::with_capacity(n - train_len);

    for (pos, &idx) in indices.iter().enumerate() {
        if pos < train_len {
            train_x.push(features[idx].clone());
            train_y.push(targets[idx]);
        } else {
            test_x.push(features[idx].clone());
            test_y.push(targets[idx]);
        }
    }

    (train_x, train_y, test_x, test_y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minmax_scaler_maps_to_unit_interval() {
        let data = vec![
            vec![1.0, 10.0],
            vec![2.0, 20.0],
            vec![3.0, 30.0],
        ];
        let scaler = MinMaxScaler::fit(&data);
        let scaled = scaler.transform(&data);

        assert!((scaled[0][0] - 0.0).abs() < 1e-12);
        assert!((scaled[1][0] - 0.5).abs() < 1e-12);
        assert!((scaled[2][0] - 1.0).abs() < 1e-12);
        assert!((scaled[1][1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn train_test_split_is_reproducible() {
        let features: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64]).collect();
        let targets: Vec<f64> = (0..10).map(|i| if i < 5 { 1.0 } else { 0.0 }).collect();

        let (tx1, ty1, vx1, vy1) = train_test_split(&features, &targets, 0.7, 42);
        let (tx2, ty2, vx2, vy2) = train_test_split(&features, &targets, 0.7, 42);

        assert_eq!(tx1.len(), 7);
        assert_eq!(vx1.len(), 3);
        assert_eq!(tx1, tx2);
        assert_eq!(ty1, ty2);
        assert_eq!(vx1, vx2);
        assert_eq!(vy1, vy2);
    }
}
