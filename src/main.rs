use ann_rust_lab07::metrics::ConfusionMatrix;
use ann_rust_lab07::nn::NeuralNet;
use ann_rust_lab07::preprocessing::{load_csv, train_test_split, MinMaxScaler};
use ann_rust_lab07::viz::export_network_svg;
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

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("iris");

    let result = match mode {
        "iris" => run_iris(),
        other => Err(format!(
            "Unknown mode '{}'. Available modes: iris (ads and uci coming in US-05/US-06)",
            other
        )),
    };

    if let Err(err) = result {
        eprintln!("Error: {}", err);
        process::exit(1);
    }
}
