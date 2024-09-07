use rand::Rng;
use serde::{Deserialize, Serialize};
//use std::ops::{Add, Mul, Sub};

mod activation;
pub mod layers;

pub type Quantized = f32;
pub type VecD1 = Vec<Quantized>; // 1D vector for dense layer
pub type VecD2 = Vec<Vec<Quantized>>; // 2D vector for convolutional layer

#[derive(Deserialize, Serialize, Clone, Debug)]
#[serde(tag = "type")]
pub enum Weights {
    Convolution { kernel: VecD2, bias: Quantized },
    Dense { weights: VecD1, bias: Quantized },
}

impl Weights {
    pub fn rand_convolution(kernel_size: usize) -> Self {
        let mut rng = rand::thread_rng();
        Weights::Convolution {
            kernel: vec![vec![rng.gen(); kernel_size]; kernel_size],
            bias: rng.gen(),
        }
    }

    pub fn rand_dense(inputs: usize) -> Self {
        let mut rng = rand::thread_rng();
        Weights::Dense {
            weights: vec![rng.gen(); inputs],
            bias: rng.gen(),
        }
    }
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU(Quantized),
    Softmax,
    None,
}

pub struct SGDOptimizer {
    learning_rate: Quantized,
}

impl SGDOptimizer {
    pub fn new(learning_rate: Quantized) -> Self {
        SGDOptimizer { learning_rate }
    }

    pub fn update(&self, weights: &mut Weights, gradients: &Weights) {
        match (weights, gradients) {
            (
                Weights::Convolution { kernel, bias },
                Weights::Convolution {
                    kernel: grad_kernel,
                    bias: grad_bias,
                },
            ) => {
                for (k, gk) in kernel.iter_mut().zip(grad_kernel.iter()) {
                    for (w, gw) in k.iter_mut().zip(gk.iter()) {
                        *w -= self.learning_rate * gw;
                    }
                }
                *bias -= self.learning_rate * grad_bias;
            }
            (
                Weights::Dense { weights, bias },
                Weights::Dense {
                    weights: grad_weights,
                    bias: grad_bias,
                },
            ) => {
                for (w, gw) in weights.iter_mut().zip(grad_weights.iter()) {
                    *w -= self.learning_rate * gw;
                }
                *bias -= self.learning_rate * grad_bias;
            }
            _ => panic!("Mismatched weight types"),
        }
    }
}

// // Implement basic arithmetic operations for Quantized type
// impl Add for Quantized {
//     type Output = Self;
//     fn add(self, other: Self) -> Self {
//         self + other
//     }
// }

// impl Sub for Quantized {
//     type Output = Self;
//     fn sub(self, other: Self) -> Self {
//         self - other
//     }
// }

// impl Mul for Quantized {
//     type Output = Self;
//     fn mul(self, other: Self) -> Self {
//         self * other
//     }
// }
