use mnist_lib::{self, MnistImage};
use model_layers::layers::{self, dense_layer, flatten_layer};
use model_layers::{Activation, Quantized, SGDOptimizer, VecD1, VecD2, Weights};
//use rayon::prelude::*;
//use std::fs::{self, File};
//use std::io::Write;
//use std::sync::atomic::{AtomicUsize, Ordering};

mod model_layers;

fn main() {
    // Load quantized weights from JSON
    let weights: Vec<Weights> = read_weights_from_json("../mnist_weights/weights.json");

    let mut model = Model::new(weights);

    // Load MNIST dataset
    let images: Vec<MnistImage> = mnist_lib::load_mnist_dataset();

    // Create optimizer
    let optimizer = SGDOptimizer::new(0.01);

    // Training loop
    for epoch in 0..10 {
        let mut total_loss = 0.0;
        for (idx, image) in images.iter().enumerate() {
            let image_data: VecD2 = image
                .data
                .iter()
                .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
                .collect();

            let mut target = vec![0.0; 10];
            target[image.label.as_usize()] = 1.0;

            let loss = model.train(image_data, target, &optimizer);
            total_loss += loss;

            if idx % 100 == 0 {
                println!("Epoch {}, Image {}: Loss = {}", epoch, idx, loss);
            }
        }
        println!(
            "Epoch {} complete. Average loss: {}",
            epoch,
            total_loss / images.len() as f32
        );
    }

    // Evaluate the model after training
    let mut correct_count = 0;
    for image in images.iter() {
        let image_data: VecD2 = image
            .data
            .iter()
            .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
            .collect();

        let output = model.forward(image_data);
        let predicted_class = get_predicted_class(output);

        if image.label == mnist_lib::MnistDigit::from_usize(predicted_class) {
            correct_count += 1;
        }
    }

    println!(
        "Accuracy after training: {:.2}%",
        correct_count as f32 / images.len() as f32 * 100.0
    );

    // Evaluate the model after training
    let mut correct_count = 0;
    for image in images.iter() {
        let image_data: VecD2 = image
            .data
            .iter()
            .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
            .collect();

        let output = model.forward(image_data);
        let predicted_class = get_predicted_class(output);

        if image.label == mnist_lib::MnistDigit::from_usize(predicted_class) {
            correct_count += 1;
        }
    }

    println!(
        "Accuracy after training: {:.2}%",
        correct_count as f32 / images.len() as f32 * 100.0
    );

    // // Load MNIST dataset
    // let images: Vec<MnistImage> = mnist_lib::load_mnist_dataset()
    //     .into_iter()
    //     .take(5)
    //     .collect();
    // let img_count = images.len();

    // // Use atomic variables for thread-safe counting
    // let correct_count = AtomicUsize::new(0);
    // let incorrect_count = AtomicUsize::new(0);

    // // Use Rayon's parallel iterator
    // //images.par_iter().enumerate().for_each(|(idx, image)| {
    // //    let image_data: VecD2 = image.data.iter()
    // //        .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
    // //        .collect();

    // images.iter().enumerate().for_each(|(idx, image)| {
    //     let image_data: VecD2 = image
    //         .data
    //         .iter()
    //         .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
    //         .collect();

    //     // Save the input image for comparison
    //     //save_to_file(format!("intermediate_output_rust_image_{}.json", idx).as_str(), serde_json::to_string_pretty(&image_data).expect("Failed to serialize image data").as_str());

    //     let predicted_result = {
    //         let mut w = weights.clone();

    //         // Convolution Layer
    //         let conv_output = model_layers::layers::convolution_layer(
    //             image_data.clone(),
    //             w.split_off(w.len() - 8),
    //             model_layers::Activation::ReLU(0.0),
    //         );
    //         //save_to_file(format!("intermediate_output_rust_conv_{}.json", idx).as_str(), serde_json::to_string_pretty(&conv_output).expect("Failed to serialize convolution output").as_str());

    //         // Flatten Layer
    //         let flatten_output = flatten_layer(conv_output.clone());
    //         save_to_file(
    //             format!("intermediate_output_rust_flatten_{}.json", idx).as_str(),
    //             serde_json::to_string_pretty(&flatten_output)
    //                 .expect("Failed to serialize flatten output")
    //                 .as_str(),
    //         );

    //         // Dense Layer
    //         let dense_output = dense_layer(
    //             flatten_output.clone(),
    //             w.split_off(w.len() - 10),
    //             model_layers::Activation::Softmax,
    //         );
    //         //save_to_file(format!("intermediate_output_rust_dense_{}.json", idx).as_str(), serde_json::to_string_pretty(&dense_output).expect("Failed to serialize dense output").as_str());

    //         print_predicted_class_raw_outputs(dense_output.clone());
    //         println!("Acutal class: {}", image.label.as_usize());

    //         get_predicted_class(dense_output)
    //     };

    //     if image.label == mnist_lib::MnistDigit::from_usize(predicted_result) {
    //         correct_count.fetch_add(1, Ordering::Relaxed);
    //     } else {
    //         incorrect_count.fetch_add(1, Ordering::Relaxed);
    //     }
    // });

    // let final_correct_count = correct_count.load(Ordering::Relaxed);
    // let final_incorrect_count = incorrect_count.load(Ordering::Relaxed);

    // println!("Total images: {}", img_count);
    // println!("Correct predictions: {}", final_correct_count);
    // println!("Incorrect predictions: {}", final_incorrect_count);
    // println!(
    //     "Accuracy: {:.2}%",
    //     final_correct_count as f32 / img_count as f32 * 100.0
    // );
}

// Structure to represent the machine learing model
pub struct Model {
    conv_weights: Vec<Weights>,
    dense_weights: Vec<Weights>,
}

impl Model {
    pub fn new(mut weights: Vec<Weights>) -> Self {
        let dense_weights = weights.split_off(weights.len() - 10);
        let conv_weights = weights.split_off(weights.len() - 8);
        Model {
            conv_weights,
            dense_weights,
        }
    }

    // pub fn new() -> Self {
    //     Model {
    //         conv_weights: vec![Weights::NewConvolution(), ],
    //         dense_weights,
    //     }
    // }

    pub fn forward(&self, input: VecD2) -> VecD1 {
        let conv_output =
            layers::convolution_layer(input, self.conv_weights.clone(), Activation::ReLU(0.0));
        let flatten_output = layers::flatten_layer(conv_output);
        layers::dense_layer(
            flatten_output,
            self.dense_weights.clone(),
            Activation::Softmax,
        )
    }

    pub fn train(&mut self, input: VecD2, target: VecD1, optimizer: &SGDOptimizer) -> Quantized {
        // Forward pass
        let conv_output = layers::convolution_layer(
            input.clone(),
            self.conv_weights.clone(),
            Activation::ReLU(0.0),
        );
        let flatten_output = flatten_layer(conv_output.clone());
        let dense_output = dense_layer(
            flatten_output.clone(),
            self.dense_weights.clone(),
            Activation::Softmax,
        );

        // Compute loss
        let loss = layers::cross_entropy_loss(&dense_output, &target);

        // Backward pass
        let (dense_gradients, flatten_grad) =
            layers::backprop_dense(&flatten_output, &dense_output, &target, &self.dense_weights);
        let conv_grad = layers::unflatten_gradient(&flatten_grad, &conv_output);
        let conv_gradients =
            layers::backprop_conv(&input, &conv_output, &conv_grad, &self.conv_weights);

        // Update weights
        for (weight, gradient) in self.dense_weights.iter_mut().zip(dense_gradients.iter()) {
            optimizer.update(weight, gradient);
        }
        for (weight, gradient) in self.conv_weights.iter_mut().zip(conv_gradients.iter()) {
            optimizer.update(weight, gradient);
        }

        loss
    }
}

// fn save_to_file(filename: &str, data: &str) {
//     fs::write(filename, data).expect("Unable to write data");
// }

fn read_weights_from_json(filename: &str) -> Vec<Weights> {
    let json_str =
        std::fs::read_to_string(filename).expect("Failed to read weights from JSON file");
    let mut weights: Vec<Weights> =
        serde_json::from_str(&json_str).expect("Failed to parse weights from JSON");
    //weights.reverse();
    weights
}

fn get_predicted_class(probabilities: VecD1) -> usize {
    probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

// fn print_predicted_class_raw_outputs(probabilities: VecD1) {
//     assert!(probabilities.len() == 10, "Invalid number of probabilities");
//     probabilities
//         .iter()
//         .enumerate()
//         .for_each(|(index, &probability)| {
//             println!("Class {}: {:.2}%", index, probability * 100.0);
//         });
// }
