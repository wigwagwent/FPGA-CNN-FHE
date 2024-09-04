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
    // TODO: Implement bias
    let (kernel, bias) = match weights {
        Weights::Convolution(kernel, bias) => (kernel, bias),
        _ => panic!("Invalid weights for convolution layer"),
    };

    let (input_height, input_width) = (input.len(), input[0].len());

    assert!(input_height < 3 || input_width < 3);

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
        Weights::Dense(weights, bias) => (weights, bias),
        _ => panic!("Invalid weights for convolution layer"),
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