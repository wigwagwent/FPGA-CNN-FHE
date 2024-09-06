use model_layers::layers::{dense_layer, flatten_layer};
use model_layers::{VecD1, VecD2, Weights};
use mnist_lib::{self, MnistImage};
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;
use std::fs::{self, File};
use std::io::Write;

mod model_layers;

fn main() {
    // Load quantized weights from JSON
    let weights: Vec<Weights> = read_weights_from_json("../mnist_weights/weights.json");

    // Load MNIST dataset
    let images: Vec<MnistImage> = mnist_lib::load_mnist_dataset().into_iter().take(5).collect();
    let img_count = images.len();

    // Use atomic variables for thread-safe counting
    let correct_count = AtomicUsize::new(0);
    let incorrect_count = AtomicUsize::new(0);

    // Use Rayon's parallel iterator
    //images.par_iter().enumerate().for_each(|(idx, image)| {
    //    let image_data: VecD2 = image.data.iter()
    //        .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
    //        .collect();

    images.iter().enumerate().for_each(|(idx, image)| {
        let image_data: VecD2 = image.data.iter()
           .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
           .collect();

        // Save the input image for comparison
        //save_to_file(format!("intermediate_output_rust_image_{}.json", idx).as_str(), serde_json::to_string_pretty(&image_data).expect("Failed to serialize image data").as_str());

        let predicted_result = {
            let mut w = weights.clone();

            // Convolution Layer
            let conv_output = model_layers::layers::convolution_layer(image_data.clone(), w.split_off(w.len() - 8), model_layers::Activation::ReLU(0.0));
            //save_to_file(format!("intermediate_output_rust_conv_{}.json", idx).as_str(), serde_json::to_string_pretty(&conv_output).expect("Failed to serialize convolution output").as_str());

            // Flatten Layer
            let flatten_output = flatten_layer(conv_output.clone());
            save_to_file(format!("intermediate_output_rust_flatten_{}.json", idx).as_str(), serde_json::to_string_pretty(&flatten_output).expect("Failed to serialize flatten output").as_str());

            // Dense Layer
            let dense_output = dense_layer(flatten_output.clone(), w.split_off(w.len() - 10), model_layers::Activation::Softmax);
            //save_to_file(format!("intermediate_output_rust_dense_{}.json", idx).as_str(), serde_json::to_string_pretty(&dense_output).expect("Failed to serialize dense output").as_str());

            print_predicted_class_raw_outputs(dense_output.clone());
            println!("Acutal class: {}", image.label.as_usize());

            get_predicted_class(dense_output)
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

fn save_to_file(filename: &str, data: &str) {
    fs::write(filename, data).expect("Unable to write data");
}

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

fn print_predicted_class_raw_outputs(probabilities: VecD1) {
    assert!(probabilities.len() == 10, "Invalid number of probabilities");
    probabilities.iter()
        .enumerate()
        .for_each(|(index, &probability)| {
            println!("Class {}: {:.2}%", index, probability * 100.0);
        });
}
