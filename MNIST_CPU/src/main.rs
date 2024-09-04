use std::fs::File;
use std::io::{self, Read};
use MNIST_LIB;

type Quantized = f32;
type VecD1 = Vec<Quantized>;
type VecD2 = Vec<Vec<Quantized>>;
type VecD3 = Vec<Vec<Vec<Quantized>>>;

type Kernel = [[Quantized; 3]; 3];

// Function to read weights from a binary file
fn read_weights(filename: &str) -> io::Result<Vec<Vec<i16>>> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let num_filters = 32; // Adjust this to match the actual number of filters
    let kernel_size = 3 * 3; // Assuming a 3x3 kernel
    let weight_size = kernel_size * std::mem::size_of::<u16>();

    if buffer.len() < weight_size * num_filters {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Buffer is too small"));
    }

    let mut weights: Vec<Vec<i16>> = Vec::new();
    let mut offset = 0;

    for _ in 0..num_filters {
        let filter: Vec<i16> = buffer[offset..offset + weight_size]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();
        weights.push(filter);
        offset += weight_size;
    }
    
    Ok(weights)
}

fn convolution_layer(input: &VecD2, kernel: Kernel) -> VecD2 {
    let (input_height, input_width) = (input.len(), input[0].len());

    assert!(input_height < 3 || input_width < 3);
    assert!(kernel.len() == 3);

    let output_height = input_height - kernel.len() + 1;
    let output_width = input_width - kernel.len() + 1;

    let mut output: VecD2 = vec![vec![0 as Quantized; output_width]; output_height];
    
    for i in 0..output_height {
        for j in 0..output_width {
            let mut sum = 0 as Quantized;
            for ki in 0..3 {
                for kj in 0..3 {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }
    relu_activation(output)
}

fn dense_layer(inputs: &VecD1, weights: VecD1, bias: Quantized) -> Quantized {
    // Ensure that the number of inputs matches the number of weights
    assert_eq!(inputs.len(), weights.len(), "Number of inputs must match the number of weights.");
    
    // Compute the weighted sum plus bias
    let mut sum = bias;
    for (input, weight) in inputs.iter().zip(weights.iter()) {
        sum += input * weight;
    }
    sum
}

fn flatten_layer(outputs: VecD3) -> VecD1 {
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

fn relu_activation (mut input: VecD2) -> VecD2 {
    for i in 0..input.len() {
        for j in 0..input[0].len() {
            input[i][j] = input[i][j].max(0 as Quantized);
        }
    }
    input
}

fn softmax_activation(scores: VecD1) -> VecD1 {
    let max_score = *scores.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let exp_scores: VecD1 = scores.iter().map(|&score| (score - max_score).exp()).collect();
    let sum_exp_scores: Quantized = exp_scores.iter().sum();
    exp_scores.iter().map(|&exp_score| exp_score / sum_exp_scores).collect()
}

fn get_predicted_class(probabilities: VecD1) -> usize {
    probabilities.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}

fn model(input: VecD2, weights: &Vec<Kernel>) -> usize {
    // Perform convolution (8 convolution layers)
    let conv_output_1 = convolution_layer(&input, weights[0]);
    let conv_output_2 = convolution_layer(&input, weights[1]);
    let conv_output_3 = convolution_layer(&input, weights[2]);
    let conv_output_4 = convolution_layer(&input, weights[3]);
    let conv_output_5 = convolution_layer(&input, weights[4]);
    let conv_output_6 = convolution_layer(&input, weights[5]);
    let conv_output_7 = convolution_layer(&input, weights[6]);
    let conv_output_8 = convolution_layer(&input, weights[7]);

    // Flatten Outputs
    let conv_output = vec![conv_output_1, conv_output_2, conv_output_3, conv_output_4, conv_output_5, conv_output_6, conv_output_7, conv_output_8];
    let flat_output = flatten_layer(conv_output);

    // Perform dense layer (10 dense layers)
    let dense_output0 = dense_layer(&flat_output, weights[8], 0.0);
    let dense_output1 = dense_layer(&flat_output, weights[8], 0.0);
    let dense_output2 = dense_layer(&flat_output, weights[8], 0.0);
    let dense_output3 = dense_layer(&flat_output, weights[8], 0.0);
    let dense_output4 = dense_layer(&flat_output, weights[8], 0.0);
    let dense_output5 = dense_layer(&flat_output, weights[8], 0.0);
    let dense_output6 = dense_layer(&flat_output, weights[8], 0.0);
    let dense_output7 = dense_layer(&flat_output, weights[8], 0.0);
    let dense_output8 = dense_layer(&flat_output, weights[8], 0.0);
    let dense_output9 = dense_layer(&flat_output, weights[8], 0.0);

    // Perform softmax activation
    let dense_outputs = vec![dense_output0, dense_output1, dense_output2, dense_output3, dense_output4, dense_output5, dense_output6, dense_output7, dense_output8, dense_output9];
    let scores = softmax_activation(dense_outputs);

    // Get the predicted digit
    get_predicted_class(scores)
}

// Main function
fn main() -> io::Result<()> {
    // Load quantized weights
    let weights = read_weights("weights.bin")?;

    // Load MNIST dataset
    let images = MNIST_LIB::load_mnist_dataset();
    let img_count = images.len();

    let mut correct_count = 0;
    let mut incorrect_count = 0;


    // Iterate through all images
    for image in images {
        let preditcted_result = model(image.data, weights);

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

    Ok(())
}
