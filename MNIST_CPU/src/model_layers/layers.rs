use crate::model_layers::activation::softmax_activation;
use crate::model_layers::activation::relu_activation;

use super::{Quantized, Activation, VecD1, VecD2, Weights};

pub fn convolution_layer(input: VecD2, weights: Vec<Weights>, activation: Activation) -> Vec<VecD2> {
    let mut output: Vec<VecD2> = Vec::new();
    for i in 0..weights.len() {
        output.push(convolution(&input, &weights[i], activation));
    }
    output
}

fn convolution(input: &VecD2, weights: &Weights, activation: Activation) -> VecD2 {
    let (kernel, bias) = match weights {
        Weights::Convolution { kernel, bias } => (kernel, bias),
        _ => panic!("Invalid weights for convolution layer"),
    };

    let (input_height, input_width) = (input.len(), input[0].len());
    let (kernel_height, kernel_width) = (kernel.len(), kernel[0].len());

    assert!(input_height >= kernel_height || input_width >= kernel_width);

    let output_height = input_height - kernel_height + 1;
    let output_width = input_width - kernel_width + 1;

    let mut output: VecD2 = vec![vec![0 as Quantized; output_width]; output_height];

    for i in 0..output_height {
        for j in 0..output_width {
            let mut sum = 0 as Quantized;
            for ki in 0..kernel_height {
                for kj in 0..kernel_width {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            // Add bias to the sum
            sum += bias;
            output[i][j] = sum;
        }
    }

    match activation {
        Activation::ReLU(min) => relu_activation(output, min),
        _ => output,
    }
}

pub fn dense_layer(inputs: VecD1, weights: Vec<Weights>, activation: Activation) -> Vec<Quantized> {
    let mut output: Vec<Quantized> = Vec::new();
    for i in 0..weights.len() {
        output.push(dense(inputs.clone(), weights[i].clone()));
    }
    match activation {
        Activation::Softmax => softmax_activation(output),
        _ => output,
    }
}

fn dense(inputs: VecD1, weights: Weights) -> Quantized {
    let (weights, bias) = match weights {
        Weights::Dense { weights, bias } => (weights, bias),
        _ => panic!("Invalid weights for dense layer"),
    };
    
    // Ensure that the number of inputs matches the number of weights
    assert_eq!(inputs.len(), weights.len(), "Number of inputs must match the number of weights.");
    
    // Compute the weighted sum plus bias
    let mut sum = bias;
    for (input, weight) in inputs.iter().zip(weights.iter()) {
        sum += input * weight;
    }
    sum
}

pub fn flatten_layer(outputs: Vec<VecD2>) -> VecD1 {
    let mut flat_output = Vec::new();
    for output in outputs {
        for row in output {
            for value in row {
                flat_output.push(value);
            }
        }
    }
    flat_output
}

#[cfg(test)]
mod tests {
    use crate::read_weights_from_json;

    use super::*;
    use serde_json::Value;
    use std::fs;

    // Helper function to load an image (5x5 or 28x28) from a JSON file
    fn load_image_from_json(file_path: &str) -> VecD2 {
        // Read the JSON file as a string
        let json_str = fs::read_to_string(file_path).expect("Failed to read image from JSON file");

        // Parse the JSON string into a 2D array of floats (VecD2)
        serde_json::from_str(&json_str).expect("Failed to parse image JSON")
    }

    fn create_mock_weights_from_json(file_path: &str) -> Weights {
        // Read the JSON file as a string
        let json_str = fs::read_to_string(file_path).expect("Failed to read weights from JSON file");
    
        // Parse the JSON string into a serde_json::Value
        let json_data: Value = serde_json::from_str(&json_str).expect("Failed to parse weights from JSON");
    
        // Extract the first weight layer, assuming it's of type "Convolution"
        let kernel = json_data[0]["kernel"]
            .as_array()
            .expect("Expected kernel array")
            .iter()
            .map(|row| {
                row.as_array()
                    .expect("Expected array of floats")
                    .iter()
                    .map(|v| v.as_f64().expect("Expected float") as f32)
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();
    
        let bias = json_data[0]["bias"].as_f64().expect("Expected bias float") as Quantized;
    
        // Create the Weights::Convolution object
        Weights::Convolution {
            kernel,
            bias,
        }
    }

    fn create_dense_weights_from_json(file_path: &str) -> Weights {
        // Read the JSON file as a string
        let json_str = fs::read_to_string(file_path).expect("Failed to read weights from JSON file");

        // Parse the JSON string into a serde_json::Value
        let json_data: Value = serde_json::from_str(&json_str).expect("Failed to parse weights from JSON");

        // Extract the dense layer weights and bias
        let weights = json_data[0]["weights"]
            .as_array()
            .expect("Expected weights array")
            .iter()
            .map(|v| v.as_f64().expect("Expected float") as f32)
            .collect::<Vec<f32>>();

        let bias = json_data[0]["bias"].as_f64().expect("Expected bias float") as Quantized;

        Weights::Dense {
            weights,
            bias,
        }
    }

    // Load the expected output from a JSON file
    fn load_expected_output_from_json(file_path: &str) -> VecD2 {
        // Read the JSON file as a string
        let json_str = fs::read_to_string(file_path).expect("Failed to read output from JSON file");

        // Parse the JSON string into a 2D array of floats
        serde_json::from_str(&json_str).expect("Failed to parse expected output JSON")
    }

    // Load the expected flattened output from a JSON file
    fn load_flattened_output_from_json(file_path: &str) -> VecD1 {
        // Read the JSON file as a string
        let json_str = fs::read_to_string(file_path).expect("Failed to read flattened output from JSON file");

        // Parse the JSON string into a 1D array of floats (VecD1)
        serde_json::from_str(&json_str).expect("Failed to parse flattened output JSON")
    }

    // Load the expected dense output from a JSON file
    fn load_dense_output_from_json(file_path: &str) -> VecD1 {
        // Read the JSON file as a string
        let json_str = fs::read_to_string(file_path).expect("Failed to read dense output from JSON file");

        // Parse the JSON string into a 1D array of floats (VecD1)
        serde_json::from_str(&json_str).expect("Failed to parse dense output JSON")
    }

    // Helper function to load the flattened input from a JSON file
    fn load_flatten_input_from_json(file_path: &str) -> VecD1 {
        // Read the JSON file as a string
        let json_str = fs::read_to_string(file_path).expect("Failed to read flattened input from JSON file");

        // Parse the JSON string into a 1D array of floats (VecD1)
        serde_json::from_str(&json_str).expect("Failed to parse flattened input JSON")
    }


    // Tolerance for floating point comparisons
    const TOLERANCE: f32 = 1e-6;

    // Test case for convolution layer
    #[test]
    fn test_convolution_layer_output_with_json() {
        // Load the image from JSON
        let image_data: VecD2 = load_image_from_json("../MNIST_WEIGHTS/test_convolution/test_image.json");
    
        // Load weights from JSON
        let weights = create_mock_weights_from_json("../MNIST_WEIGHTS/test_convolution/test_weights.json");
    
        // Apply the convolution layer to the image data
        let conv_output = convolution(&image_data, &weights, Activation::None);
    
        // Load expected output from JSON
        let expected_output = load_expected_output_from_json("../MNIST_WEIGHTS/test_convolution/output.json");
    
        // Check if the output matches the expected output with a tolerance
        for (i, (row_actual, row_expected)) in conv_output.iter().zip(expected_output.iter()).enumerate() {
            for (j, (val_actual, val_expected)) in row_actual.iter().zip(row_expected.iter()).enumerate() {
                if (val_actual - val_expected).abs() > TOLERANCE {
                    println!(
                        "Difference at position ({}, {}): Actual = {}, Expected = {}",
                        i, j, val_actual, val_expected
                    );
                    panic!("Test failed: Output values differ by more than tolerance at position ({}, {})", i, j);
                }
            }
        }
        println!("Test passed: Convolution output matches expected output.");
    }
    
    // Test case for ReLU activation with convolution layer
    #[test]
    fn test_relu_convolution_layer_output_with_json() {
        // Load the image from JSON
        let image_data: VecD2 = load_image_from_json("../MNIST_WEIGHTS/test_relu/test_image.json");

        // Load weights from JSON
        let weights = create_mock_weights_from_json("../MNIST_WEIGHTS/test_relu/test_weights.json");

        // Apply the convolution layer to the image data
        let conv_output = convolution(&image_data, &weights, Activation::ReLU(0.0)); // Apply ReLU activation

        // Load expected output from JSON
        let expected_output = load_expected_output_from_json("../MNIST_WEIGHTS/test_relu/output.json");

        // Check if the output matches the expected output with a tolerance
        for (i, (row_actual, row_expected)) in conv_output.iter().zip(expected_output.iter()).enumerate() {
            for (j, (val_actual, val_expected)) in row_actual.iter().zip(row_expected.iter()).enumerate() {
                if (val_actual - val_expected).abs() > TOLERANCE {
                    println!(
                        "Difference at position ({}, {}): Actual = {}, Expected = {}",
                        i, j, val_actual, val_expected
                    );
                    panic!("Test failed: Output values differ by more than tolerance at position ({}, {})", i, j);
                }
            }
        }

        println!("Test passed: ReLU output matches expected output.");
    }

    // Test case for flatten layer
    #[test]
    fn test_flatten_layer_output_with_json() {
        // Load the output from the convolution layer (5x5), which was generated from a previous step
        let conv_output: Vec<VecD2> = vec![load_image_from_json("../MNIST_WEIGHTS/test_flatten/test_image.json")];

        // Apply the flatten layer
        let flatten_output = flatten_layer(conv_output);

        // Load the expected flattened output from JSON
        let expected_flattened_output = load_flattened_output_from_json("../MNIST_WEIGHTS/test_flatten/flatten_output.json");

        // Check if the flattened output matches the expected output
        assert_eq!(flatten_output, expected_flattened_output, "Flatten layer output does not match expected output.");
        println!("Test passed: Flatten output matches expected output.");
    }

    // Test case for dense layer
    #[test]
    fn test_dense_layer_output_with_json() {
        // Load the flattened input from JSON
        let flattened_input: VecD1 = load_flattened_output_from_json("../MNIST_WEIGHTS/test_dense/flatten_input.json");

        // Load weights from JSON
        let dense_weights = create_dense_weights_from_json("../MNIST_WEIGHTS/test_dense/test_weights.json");

        // Apply the dense layer to the flattened input
        let dense_output = dense(flattened_input, dense_weights);

        // Load the expected dense layer output from JSON
        let expected_dense_output = load_dense_output_from_json("../MNIST_WEIGHTS/test_dense/dense_output.json");

        // Check if the dense layer output matches the expected output
        assert_eq!(dense_output, expected_dense_output[0], "Dense layer output does not match expected output.");
        println!("Test passed: Dense layer output matches expected output.");
    }

    // Helper function to load softmax output from JSON
    fn load_softmax_output_from_json(file_path: &str) -> VecD1 {
        // Read the JSON file as a string
        let json_str = fs::read_to_string(file_path).expect("Failed to read softmax output from JSON file");

        // Parse the JSON string into a 1D array of floats (VecD1)
        serde_json::from_str(&json_str).expect("Failed to parse softmax output JSON")
    }

    // Test case for softmax activation
    #[test]
    fn test_softmax_layer_output_with_json() {
        // Load the flattened input from JSON (same as before)
        let flatten_input: VecD1 = load_flatten_input_from_json("../mnist_weights/test_softmax/flatten_input.json");

        // Load the weights from JSON
        let dense_weights = create_dense_weights_from_json("../mnist_weights/test_softmax/test_weights.json");

        // Apply the dense layer
        let dense_output = dense_layer(flatten_input, vec![dense_weights], Activation::None);

        // Apply softmax to the dense layer output
        let softmax_output = softmax_activation(dense_output);

        // Load the expected softmax output from JSON
        let expected_softmax_output = load_softmax_output_from_json("../mnist_weights/test_softmax/softmax_output.json");

        // Check if the softmax output matches the expected output
        assert_eq!(softmax_output, expected_softmax_output, "Softmax layer output does not match expected output.");
    }

    #[test]
    fn test_convolution_to_flatten_layer() {
        let image_data: VecD2 = load_image_from_json("../mnist_weights/test_convolution_to_flatten/test_image.json");
        let mut w: Vec<Weights> = read_weights_from_json("../mnist_weights/weights.json");

            // Convolution Layer
            let conv_output = convolution_layer(image_data.clone(), w.split_off(w.len() - 8), Activation::ReLU(0.0));
            //save_to_file(format!("intermediate_output_rust_conv_{}.json", idx).as_str(), serde_json::to_string_pretty(&conv_output).expect("Failed to serialize convolution output").as_str());

            // Flatten Layer
            let flatten_output = flatten_layer(conv_output.clone());
            //save_to_file(format!("intermediate_output_rust_flatten_{}.json", idx).as_str(), serde_json::to_string_pretty(&flatten_output).expect("Failed to serialize flatten output").as_str());

            // Load the expected flatten output from JSON
            let expected_flatten_output: VecD1 = load_flattened_output_from_json("../mnist_weights/test_convolution_to_flatten/output.json");

            // Compare the flatten output with the expected output
            for (actual, expected) in flatten_output.iter().zip(expected_flatten_output.iter()) {
                assert!(
                    (actual - expected).abs() < TOLERANCE,
                    "Flatten output does not match expected output."
                );
            }
            println!("Test passed: Flatten output matches expected output.");

    }
}
