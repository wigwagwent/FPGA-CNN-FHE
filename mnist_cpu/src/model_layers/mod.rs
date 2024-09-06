use serde::{Deserialize, Serialize};

pub mod layers;
mod activation;

pub type Quantized = f32;
pub type VecD1 = Vec<Quantized>;  // 1D vector for dense layer
pub type VecD2 = Vec<Vec<Quantized>>; // 2D vector for convolutional layer

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum Weights {
    Convolution {
        kernel: VecD2,
        bias: Quantized,
    },
    Dense {
        weights: VecD1,
        bias: Quantized,
    },
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU(Quantized),
    Softmax,
    None,
}