use ann_rust_lab07::metrics::ConfusionMatrix;
use ann_rust_lab07::nn::NeuralNet;
use ann_rust_lab07::preprocessing::{load_csv, train_test_split, MinMaxScaler};

#[test]
fn iris_pipeline_classifies_setosa_accurately() {
    let dataset = load_csv("data/iris.csv", &[0, 1, 2, 3], 4, "setosa")
        .expect("data/iris.csv should load");
    let scaler = MinMaxScaler::fit(&dataset.features);
    let scaled = scaler.transform(&dataset.features);
    let (train_x, train_y, test_x, test_y) = train_test_split(&scaled, &dataset.targets, 0.7, 42);

    let mut nn = NeuralNet::new(&[4, 3, 1], 0.3, 42);
    nn.fit(&train_x, &train_y, 2_000, 0.009);

    let probs = nn.predict_batch(&test_x);
    let cm = ConfusionMatrix::from_predictions(&test_y, &probs, 0.5);
    assert!(
        cm.accuracy() >= 0.95,
        "Expected >= 0.95 accuracy on Iris test set, got {}",
        cm.accuracy()
    );

    let new_scaled = scaler.transform_row(&[5.0, 3.4, 1.5, 0.2]);
    let prob = nn.compute(&new_scaled);
    assert!(prob > 0.9, "Expected Setosa probability > 0.9, got {}", prob);
}

#[test]
fn uci_breast_cancer_pipeline_achieves_high_accuracy() {
    let feature_indices: Vec<usize> = (0..30).collect();
    let dataset = load_csv("data/uci_breast_cancer.csv", &feature_indices, 30, "malignant")
        .expect("data/uci_breast_cancer.csv should load");
    let scaler = MinMaxScaler::fit(&dataset.features);
    let scaled = scaler.transform(&dataset.features);
    let (train_x, train_y, test_x, test_y) = train_test_split(&scaled, &dataset.targets, 0.7, 42);

    let mut nn = NeuralNet::new(&[30, 4, 1], 0.2, 42);
    nn.fit(&train_x, &train_y, 1_500, 0.5);

    let probs = nn.predict_batch(&test_x);
    let cm = ConfusionMatrix::from_predictions(&test_y, &probs, 0.5);
    assert!(
        cm.accuracy() >= 0.92,
        "Expected >= 0.92 accuracy on UCI Breast Cancer test set, got {}",
        cm.accuracy()
    );
    assert_eq!(nn.input_feature_importance().len(), 30);
}
