//! SVG visualization of a trained feed-forward NeuralNet (equivalent to R `plot(nn)`).

use crate::nn::NeuralNet;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

/// Exports an SVG diagram showing layers, nodes, connection weights, and bias terms.
pub fn export_network_svg<P: AsRef<Path>>(
    nn: &NeuralNet,
    feature_names: &[String],
    output_name: &str,
    out_path: P,
) -> Result<(), String> {
    if let Some(parent) = out_path.as_ref().parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create output directory {:?}: {}", parent, e))?;
    }

    let width = 920.0_f64;
    let height = 560.0_f64;
    let num_layers = nn.layer_sizes.len();
    let x_margin = 160.0_f64;
    let x_step = if num_layers > 1 {
        (width - 2.0 * x_margin) / (num_layers - 1) as f64
    } else {
        0.0
    };

    // Compute (x, y) coordinates for every node in every layer
    let mut coords: Vec<Vec<(f64, f64)>> = Vec::with_capacity(num_layers);
    for (l_idx, &count) in nn.layer_sizes.iter().enumerate() {
        let x = x_margin + l_idx as f64 * x_step;
        let top_pad = 90.0;
        let usable_h = height - 170.0;
        let mut layer_coords = Vec::with_capacity(count);
        for n_idx in 0..count {
            let y = if count == 1 {
                top_pad + usable_h / 2.0
            } else {
                top_pad + (n_idx as f64 / (count - 1) as f64) * usable_h
            };
            layer_coords.push((x, y));
        }
        coords.push(layer_coords);
    }

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        width, height, width, height
    ));
    svg.push('\n');
    svg.push_str(r#"  <rect width="100%" height="100%" fill="#ffffff"/>"#);
    svg.push('\n');

    // Draw weighted edges
    for (l_idx, layer) in nn.layers.iter().enumerate() {
        for (j, w_row) in layer.weights.iter().enumerate() {
            let (x2, y2) = coords[l_idx + 1][j];
            for (i, &w) in w_row.iter().enumerate() {
                let (x1, y1) = coords[l_idx][i];
                let color = if w >= 0.0 { "#2563eb" } else { "#dc2626" };
                let stroke_w = (w.abs() * 0.6).clamp(0.8, 3.0);
                svg.push_str(&format!(
                    r#"  <line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="{}" stroke-width="{:.2}" stroke-opacity="0.75"/>"#,
                    x1, y1, x2, y2, color, stroke_w
                ));
                svg.push('\n');

                let mx = x1 + (x2 - x1) * 0.42;
                let my = y1 + (y2 - y1) * 0.42 - 4.0;
                svg.push_str(&format!(
                    r#"  <text x="{:.1}" y="{:.1}" font-family="monospace" font-size="10" fill="#1f2937" text-anchor="middle">{:.3}</text>"#,
                    mx, my, w
                ));
                svg.push('\n');
            }
        }
    }

    // Draw bias nodes and bias edges
    for (l_idx, layer) in nn.layers.iter().enumerate() {
        let bx = (coords[l_idx][0].0 + coords[l_idx + 1][0].0) / 2.0;
        let by = 36.0;
        svg.push_str(&format!(
            r#"  <circle cx="{:.1}" cy="{:.1}" r="14" fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.5"/>"#,
            bx, by
        ));
        svg.push('\n');
        svg.push_str(&format!(
            r#"  <text x="{:.1}" y="{:.1}" font-family="sans-serif" font-size="12" fill="#1e3a8a" text-anchor="middle">1</text>"#,
            bx, by + 4.0
        ));
        svg.push('\n');

        for (j, &b) in layer.biases.iter().enumerate() {
            let (x2, y2) = coords[l_idx + 1][j];
            svg.push_str(&format!(
                r#"  <line x1="{:.1}" y1="{:.1}" x2="{:.1}" y2="{:.1}" stroke="#3b82f6" stroke-width="1.0" stroke-dasharray="4,3"/>"#,
                bx, by + 14.0, x2, y2
            ));
            svg.push('\n');
            let mx = bx + (x2 - bx) * 0.55;
            let my = by + (y2 - by) * 0.55;
            svg.push_str(&format!(
                r#"  <text x="{:.1}" y="{:.1}" font-family="monospace" font-size="10" fill="#1d4ed8">{:.3}</text>"#,
                mx, my, b
            ));
            svg.push('\n');
        }
    }

    // Draw neurons and labels
    for (l_idx, layer_coords) in coords.iter().enumerate() {
        for (n_idx, &(x, y)) in layer_coords.iter().enumerate() {
            let fill = if l_idx == 0 {
                "#f3f4f6"
            } else if l_idx == num_layers - 1 {
                "#dcfce7"
            } else {
                "#fef3c7"
            };
            svg.push_str(&format!(
                r#"  <circle cx="{:.1}" cy="{:.1}" r="16" fill="{}" stroke="#111827" stroke-width="1.8"/>"#,
                x, y, fill
            ));
            svg.push('\n');

            if l_idx == 0 {
                let label = feature_names
                    .get(n_idx)
                    .map(String::as_str)
                    .unwrap_or("Input");
                svg.push_str(&format!(
                    r#"  <text x="{:.1}" y="{:.1}" font-family="sans-serif" font-size="13" fill="#111827" text-anchor="end">{}</text>"#,
                    x - 24.0, y + 4.0, label
                ));
                svg.push('\n');
            } else if l_idx == num_layers - 1 && n_idx == 0 {
                svg.push_str(&format!(
                    r#"  <text x="{:.1}" y="{:.1}" font-family="sans-serif" font-size="13" fill="#111827" text-anchor="start">{}</text>"#,
                    x + 24.0, y + 4.0, output_name
                ));
                svg.push('\n');
            }
        }
    }

    // Footer with Error and Steps
    svg.push_str(&format!(
        r#"  <text x="{:.1}" y="{:.1}" font-family="sans-serif" font-size="14" fill="#111827" text-anchor="middle">Error: {:.5}   Steps: {}</text>"#,
        width / 2.0,
        height - 20.0,
        nn.final_error,
        nn.steps
    ));
    svg.push('\n');
    svg.push_str("</svg>\n");

    let mut file = File::create(out_path.as_ref())
        .map_err(|e| format!("Failed to create SVG {:?}: {}", out_path.as_ref(), e))?;
    file.write_all(svg.as_bytes())
        .map_err(|e| format!("Failed to write SVG: {}", e))?;

    Ok(())
}
