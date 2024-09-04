use std::fs::File;
use std::io::Read;
use model_layers::layers::{dense_layer, flatten_layer};
use crate::model_layers::KernelExt;
use model_layers::{Kernel, VecD1, VecD2, Weights};
use MNIST_LIB;

mod model_layers;

fn main() {
    // Load quantized weights
    let weights: Vec<Weights> = read_weights("weights.bin");

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

        let preditcted_result = {
            let mut w = weights.clone();
            let model = model_layers::layers::convolution_layer(image_data, w.split_off(w.len() - 8), model_layers::Activation::ReLU(0.0));
            let model = flatten_layer(model);
            let model = dense_layer(model, w.split_off(w.len() - 10), model_layers::Activation::Softmax);
            get_predicted_class(model)
        };

        if image.label == MNIST_LIB::MnistDigit::from_usize(preditcted_result) {
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

// Function to read weights from a binary file
fn read_weights(filename: &str) -> Vec<Weights> {
    let mut file = File::open(filename).expect("Failed to open file");
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).expect("Failed to read file");

    let kernel_size = Kernel::kernel_size();

    let mut weights: Vec<Weights> = Vec::new();
    for _ in 0..8 {
        weights.push(Weights::Convolution(
            [
                [0.0; 3],
                [0.0; 3],
                [0.0; 3],
            ],
            0.0,
        ));
    }
    for _ in 0..10 {
        weights.push(Weights::Dense(
            [0.0; 10].to_vec(),
            0.0,
        ));
    }
    weights.reserve(0); // Reverse to use split_off
    weights
}


fn get_predicted_class(probabilities: VecD1) -> usize {
    probabilities.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}