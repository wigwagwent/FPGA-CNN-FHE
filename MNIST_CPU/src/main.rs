use std::fs::File;
use std::io::BufReader;
use serde::{Deserialize};
use serde_json::from_reader;
use model_layers::layers::{dense_layer, flatten_layer};
use crate::model_layers::KernelExt;
use model_layers::{Kernel, VecD1, VecD2, Weights};
use MNIST_LIB;

mod model_layers;
// Structs to represent weights loaded from JSON
#[derive(Deserialize)]
struct WeightData {
    conv_weights: Vec<Vec<Vec<Vec<f32>>>>,
    conv_bias: VecD1, // Changed from f32 to Vec<f32>
    dense_weights: VecD2,
    dense_bias: VecD1,
}

fn main() {
    // Load quantized weights from JSON
    let weights: Vec<Weights> = read_weights_from_json("weights.json");

    // Load MNIST dataset
    let images = MNIST_LIB::load_mnist_dataset();
    let img_count = images.len();

    let mut correct_count = 0;
    let mut incorrect_count = 0;

    // Iterate through all images
    for image in images {
        let image_data: VecD2 = image.data.iter()
            .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
            .collect();

        let predicted_result = {
            let mut w = weights.clone();
            let model = model_layers::layers::convolution_layer(image_data, w.split_off(w.len() - 8), model_layers::Activation::ReLU(0.0));
            let model = flatten_layer(model);
            let model = dense_layer(model, w.split_off(w.len() - 10), model_layers::Activation::Softmax);
            get_predicted_class(model)
        };

        if image.label == MNIST_LIB::MnistDigit::from_usize(predicted_result) {
            correct_count += 1;
        } else {
            incorrect_count += 1;
        }
    }

    println!("Total images: {}", img_count);
    println!("Correct predictions: {}", correct_count);
    println!("Incorrect predictions: {}", incorrect_count);
    println!("Accuracy: {:.2}%", correct_count as f32 / img_count as f32 * 100.0);
}

fn read_weights_from_json(filename: &str) -> Vec<Weights> {
    let file = File::open(filename).expect("Failed to open JSON file");
    let reader = BufReader::new(file);

    // Parse the JSON into a vector of WeightData structs
    let weight_data: Vec<WeightData> = from_reader(reader).expect("Failed to parse JSON");

    // Convert the parsed JSON data into the appropriate Weights enum structure
    let mut weights: Vec<Weights> = Vec::new();

    // Assuming first 8 layers are convolution layers and remaining are dense layers
    // for w in weight_data.iter().take(8) {
    //     let kernels = Kernel::new(w.conv_weights.clone()); // Handle multiple kernels
    //     for kernel in kernels {
    //         weights.push(Weights::Convolution(kernel, w.conv_bias));
    //     }
    // }

    // for w in weight_data.iter().skip(8) {
    //     weights.push(Weights::Dense(w.dense_weights.clone(), w.dense_bias));
    // }

    weights
}

fn get_predicted_class(probabilities: VecD1) -> usize {
    probabilities.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}