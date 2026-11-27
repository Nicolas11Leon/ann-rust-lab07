use ann_rust_lab07::metrics::ConfusionMatrix;
use ann_rust_lab07::nn::NeuralNet;
use ann_rust_lab07::preprocessing::{load_csv, train_test_split, MinMaxScaler};
use ann_rust_lab07::viz::export_network_svg;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::svm::svc::{SVCParameters, SVC};
use smartcore::svm::Kernels;
use smartcore::tree::decision_tree_classifier::DecisionTreeClassifier;
use std::env;
use std::process;

fn run_iris() -> Result<(), String> {
    let dataset = load_csv("data/iris.csv", &[0, 1, 2, 3], 4, "setosa")?;
    let scaler = MinMaxScaler::fit(&dataset.features);
    let scaled = scaler.transform(&dataset.features);

    let (train_x, train_y, test_x, test_y) = train_test_split(&scaled, &dataset.targets, 0.7, 42);

    // One hidden layer of 3 units (matching Listing 3: hidden = 3)
    let mut nn = NeuralNet::new(&[4, 3, 1], 0.3, 42);
    nn.fit(&train_x, &train_y, 2_000, 0.009);

    println!(
        "Training finished — Error: {:.5}, Steps: {}",
        nn.final_error, nn.steps
    );

    export_network_svg(&nn, &dataset.feature_names, "is_setosa", "outputs/iris_nn.svg")?;
    println!("Saved network architecture diagram to outputs/iris_nn.svg");

    let probs = nn.predict_batch(&test_x);
    let cm = ConfusionMatrix::from_predictions(&test_y, &probs, 0.5);
    cm.print_report("Iris Test Set (is_setosa)");

    // Listing 6: New tuple SL=5.0, SW=3.4, PL=1.5, PW=0.2
    let new_tuple = [5.0, 3.4, 1.5, 0.2];
    let new_scaled = scaler.transform_row(&new_tuple);
    let prob_setosa = nn.compute(&new_scaled);
    println!("Probability of Setosa (5.0, 3.4, 1.5, 0.2): {:.4}", prob_setosa);

    Ok(())
}

fn to_dense_matrix(rows: &[Vec<f64>]) -> DenseMatrix<f64> {
    let refs: Vec<&[f64]> = rows.iter().map(Vec::as_slice).collect();
    DenseMatrix::from_2d_array(&refs).expect("Failed to build DenseMatrix")
}

fn run_ads() -> Result<(), String> {
    // Lab 04 Section 3: Advertisement dataset (Age, EstimatedSalary -> Purchased), 75/25 split
    let dataset = load_csv("data/advertisement.csv", &[2, 3], 4, "1")?;
    let (raw_train_x, train_y, raw_test_x, test_y) =
        train_test_split(&dataset.features, &dataset.targets, 0.75, 42);

    let scaler = MinMaxScaler::fit(&raw_train_x);
    let train_x = scaler.transform(&raw_train_x);
    let test_x = scaler.transform(&raw_test_x);

    // 1. Artificial Neural Network (Pure Rust MLP)
    let mut nn = NeuralNet::new(&[2, 4, 1], 0.3, 42);
    nn.fit(&train_x, &train_y, 2_000, 1.0);
    println!(
        "Advertisement ANN Training finished — Error: {:.5}, Steps: {}",
        nn.final_error, nn.steps
    );
    export_network_svg(&nn, &dataset.feature_names, "Purchased", "outputs/ads_nn.svg")?;
    println!("Saved network architecture diagram to outputs/ads_nn.svg");

    let ann_probs = nn.predict_batch(&test_x);
    let ann_cm = ConfusionMatrix::from_predictions(&test_y, &ann_probs, 0.5);
    ann_cm.print_report("ANN (Pure Rust MLP) — Advertisement Test Set");

    // 2. SmartCore Decision Tree (Lab 04 Autonomous Work)
    let x_train_dm = to_dense_matrix(&train_x);
    let x_test_dm = to_dense_matrix(&test_x);
    let y_train_u32: Vec<u32> = train_y.iter().map(|&y| if y >= 0.5 { 1 } else { 0 }).collect();
    let y_train_i32: Vec<i32> = train_y.iter().map(|&y| if y >= 0.5 { 1 } else { 0 }).collect();

    let dt = DecisionTreeClassifier::fit(&x_train_dm, &y_train_u32, Default::default())
        .map_err(|e| format!("DecisionTree fit error: {}", e))?;
    let dt_preds = dt
        .predict(&x_test_dm)
        .map_err(|e| format!("DecisionTree predict error: {}", e))?;
    let dt_probs: Vec<f64> = dt_preds.into_iter().map(|v| v as f64).collect();
    let dt_cm = ConfusionMatrix::from_predictions(&test_y, &dt_probs, 0.5);
    dt_cm.print_report("Decision Tree (SmartCore) — Advertisement Test Set");

    // 3. SmartCore SVM / SVC (RBF Kernel)
    let svc_params = SVCParameters::default()
        .with_kernel(Kernels::rbf().with_gamma(1.0))
        .with_c(1.0);
    let svc = SVC::fit(&x_train_dm, &y_train_i32, &svc_params)
        .map_err(|e| format!("SVC fit error: {}", e))?;
    let svc_preds = svc
        .predict(&x_test_dm)
        .map_err(|e| format!("SVC predict error: {}", e))?;
    let svc_probs: Vec<f64> = svc_preds.into_iter().map(|v| if v > 0.0 { 1.0 } else { 0.0 }).collect();
    let svc_cm = ConfusionMatrix::from_predictions(&test_y, &svc_probs, 0.5);
    svc_cm.print_report("SVM / SVC (SmartCore) — Advertisement Test Set");

    println!("\n=== Model Comparison Summary (Advertisement) ===");
    println!("{:<28} {:>10} {:>10} {:>10} {:>10}", "Model", "Accuracy", "Precision", "Recall", "F1-Score");
    for (name, cm) in [
        ("ANN (Pure Rust MLP)", ann_cm),
        ("Decision Tree (SmartCore)", dt_cm),
        ("SVM / SVC (SmartCore)", svc_cm),
    ] {
        println!(
            "{:<28} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            name,
            cm.accuracy(),
            cm.precision(),
            cm.recall(),
            cm.f1_score()
        );
    }

    Ok(())
}

fn run_uci() -> Result<(), String> {
    let feature_indices: Vec<usize> = (0..30).collect();
    let dataset = load_csv("data/uci_breast_cancer.csv", &feature_indices, 30, "malignant")?;
    let scaler = MinMaxScaler::fit(&dataset.features);
    let scaled = scaler.transform(&dataset.features);

    let (train_x, train_y, test_x, test_y) = train_test_split(&scaled, &dataset.targets, 0.7, 42);

    // Interpretable ANN: 30 inputs -> 4 hidden units -> 1 output (malignant = 1)
    let mut nn = NeuralNet::new(&[30, 4, 1], 0.2, 42);
    nn.fit(&train_x, &train_y, 1_500, 0.5);

    println!(
        "UCI Breast Cancer Training finished — Error: {:.5}, Steps: {}",
        nn.final_error, nn.steps
    );

    export_network_svg(&nn, &dataset.feature_names, "is_malignant", "outputs/uci_nn.svg")?;
    println!("Saved network architecture diagram to outputs/uci_nn.svg");

    let probs = nn.predict_batch(&test_x);
    let cm = ConfusionMatrix::from_predictions(&test_y, &probs, 0.5);
    cm.print_report("UCI Breast Cancer Wisconsin (Test Set)");

    // Interpretability: Top 5 most influential features by |input weights|
    let importances = nn.input_feature_importance();
    let mut ranked: Vec<(&str, f64)> = dataset
        .feature_names
        .iter()
        .map(String::as_str)
        .zip(importances)
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 Most Influential Features (sum |w|):");
    for (rank, (name, score)) in ranked.iter().take(5).enumerate() {
        println!("  {}. {:<28} {:.4}", rank + 1, name, score);
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("iris");

    let result = match mode {
        "iris" => run_iris(),
        "ads" => run_ads(),
        "uci" => run_uci(),
        other => Err(format!(
            "Unknown mode '{}'. Available modes: iris, ads, uci",
            other
        )),
    };

    if let Err(err) = result {
        eprintln!("Error: {}", err);
        process::exit(1);
    }
}
