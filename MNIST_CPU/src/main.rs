use model_layers::layers::{dense_layer, flatten_layer};
use model_layers::{VecD1, VecD2, Weights};
use mnist_lib;
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;

mod model_layers;


fn main() {
    // Load quantized weights from JSON
    let weights: Vec<Weights> = read_weights_from_json("../mnist_weights/weights.json");

    // Load MNIST dataset
    let images = mnist_lib::load_mnist_dataset();
    let img_count = images.len();

    // Use atomic variables for thread-safe counting
    let correct_count = AtomicUsize::new(0);
    let incorrect_count = AtomicUsize::new(0);

    // Use Rayon's parallel iterator
    images.par_iter().for_each(|image| {
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

        if image.label == mnist_lib::MnistDigit::from_usize(predicted_result) {
            correct_count.fetch_add(1, Ordering::Relaxed);
        } else {
            incorrect_count.fetch_add(1, Ordering::Relaxed);
        }
    });

    let final_correct_count = correct_count.load(Ordering::Relaxed);
    let final_incorrect_count = incorrect_count.load(Ordering::Relaxed);

    println!("Total images: {}", img_count);
    println!("Correct predictions: {}", final_correct_count);
    println!("Incorrect predictions: {}", final_incorrect_count);
    println!("Accuracy: {:.2}%", final_correct_count as f32 / img_count as f32 * 100.0);
}

// fn main() {
//     // Load quantized weights from JSON
//     let weights: Vec<Weights> = read_weights_from_json("../mnist_weights/weights.json");

//     // Load MNIST dataset
//     let images = mnist_lib::load_mnist_dataset();
//     let img_count = images.len();

//     let mut correct_count = 0;
//     let mut incorrect_count = 0;

//     // Iterate through all images
//     for image in images {
//         let image_data: VecD2 = image.data.iter()
//             .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
//             .collect();

//         let predicted_result = {
//             let mut w = weights.clone();
//             let model = model_layers::layers::convolution_layer(image_data, w.split_off(w.len() - 8), model_layers::Activation::ReLU(0.0));
//             let model = flatten_layer(model);
//             let model = dense_layer(model, w.split_off(w.len() - 10), model_layers::Activation::Softmax);
//             get_predicted_class(model)
//         };

//         if image.label == mnist_lib::MnistDigit::from_usize(predicted_result) {
//             correct_count += 1;
//         } else {
//             incorrect_count += 1;
//         }
//     }

//     println!("Total images: {}", img_count);
//     println!("Correct predictions: {}", correct_count);
//     println!("Incorrect predictions: {}", incorrect_count);
//     println!("Accuracy: {:.2}%", correct_count as f32 / img_count as f32 * 100.0);
// }


fn read_weights_from_json(filename: &str) -> Vec<Weights> {
    let json_str = std::fs::read_to_string(filename).expect("Failed to read weights from JSON file");
    let mut weights: Vec<Weights> = serde_json::from_str(&json_str).expect("Failed to parse weights from JSON");
    weights.reverse();
    weights
}

fn get_predicted_class(probabilities: VecD1) -> usize {
    probabilities.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}