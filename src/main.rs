use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mode = args.get(1).map(String::as_str).unwrap_or("iris");

    println!(
        "ann-rust-lab07 v{} — selected mode: {}",
        ann_rust_lab07::crate_version(),
        mode
    );
}
