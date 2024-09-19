use fhe_ckks::Plaintext;
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

#[derive(Clone, Debug)]
pub enum WeightsFhe {
    Convolution { kernel: VecD2, bias: Quantized },
    Dense { weights: Plaintext, bias: Plaintext },
}

impl From<Weights> for WeightsFhe {
    fn from(weights: Weights) -> Self {
        match weights {
            Weights::Convolution { kernel, bias } => WeightsFhe::Convolution { kernel, bias },
            Weights::Dense { weights, bias } => WeightsFhe::Dense {
                weights: Plaintext::from(weights.clone()),
                bias: Plaintext::from(vec![bias; weights.len()]),
            },
        }
    }
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
    Quadratic,
    Softmax,
    None,
}

pub struct SGDOptimizer {
    learning_rate: Quantized,
    momentum: Quantized,            // Momentum factor
    dense_velocities: Vec<Weights>, // Velocities for dense layers
    conv_velocities: Vec<Weights>,  // Velocities for convolution layers
}

impl SGDOptimizer {
    pub fn new(learning_rate: Quantized) -> Self {
        SGDOptimizer {
            learning_rate,
            momentum: 0.9, // Default momentum value
            dense_velocities: Vec::new(),
            conv_velocities: Vec::new(),
        }
    }

    // Initialize the velocities for both dense and convolutional weights
    pub fn initialize_velocities(
        &mut self,
        dense_weights: &Vec<Weights>,
        conv_weights: &Vec<Weights>,
    ) {
        self.dense_velocities = dense_weights
            .iter()
            .map(|w| match w {
                Weights::Dense { weights, bias: _ } => Weights::Dense {
                    weights: vec![0.0; weights.len()],
                    bias: 0.0,
                },
                _ => panic!("Unexpected weight type for dense layer"),
            })
            .collect();

        self.conv_velocities = conv_weights
            .iter()
            .map(|w| match w {
                Weights::Convolution { kernel, bias: _ } => Weights::Convolution {
                    kernel: vec![vec![0.0; kernel[0].len()]; kernel.len()],
                    bias: 0.0,
                },
                _ => panic!("Unexpected weight type for convolution layer"),
            })
            .collect();
    }

    // Update the weights using momentum and the gradients
    pub fn update(&mut self, weights: &mut Weights, gradients: &Weights) {
        match (weights, gradients) {
            (
                Weights::Convolution { kernel, bias },
                Weights::Convolution {
                    kernel: grad_kernel,
                    bias: grad_bias,
                },
            ) => {
                // Ensure velocities are initialized
                if self.conv_velocities.is_empty() {
                    panic!("Conv velocities not initialized.");
                }

                // Get the corresponding velocity for this weight
                let velocity = self
                    .conv_velocities
                    .iter_mut()
                    .find(|v| matches!(v, Weights::Convolution { .. }))
                    .unwrap();

                match velocity {
                    Weights::Convolution {
                        kernel: vel_kernel,
                        bias: vel_bias,
                    } => {
                        // Update the velocities and weights for convolutional layers
                        for ((v_k, g_k), w_k) in vel_kernel
                            .iter_mut()
                            .zip(grad_kernel.iter())
                            .zip(kernel.iter_mut())
                        {
                            for ((v, g), w) in v_k.iter_mut().zip(g_k.iter()).zip(w_k.iter_mut()) {
                                *v = self.momentum * (*v) + (1.0 - self.momentum) * g;
                                *w -= self.learning_rate * (*v);
                            }
                        }
                        *vel_bias = self.momentum * (*vel_bias) + (1.0 - self.momentum) * grad_bias;
                        *bias -= self.learning_rate * (*vel_bias);
                    }
                    _ => panic!("Unexpected velocity type for convolution layer"),
                }
            }
            (
                Weights::Dense { weights, bias },
                Weights::Dense {
                    weights: grad_weights,
                    bias: grad_bias,
                },
            ) => {
                // Ensure velocities are initialized
                if self.dense_velocities.is_empty() {
                    panic!("Dense velocities not initialized.");
                }

                // Get the corresponding velocity for this weight
                let velocity = self
                    .dense_velocities
                    .iter_mut()
                    .find(|v| matches!(v, Weights::Dense { .. }))
                    .unwrap();

                match velocity {
                    Weights::Dense {
                        weights: vel_weights,
                        bias: vel_bias,
                    } => {
                        // Update the velocities and weights for dense layers
                        for ((v_w, g_w), w_w) in vel_weights
                            .iter_mut()
                            .zip(grad_weights.iter())
                            .zip(weights.iter_mut())
                        {
                            *v_w = self.momentum * (*v_w) + (1.0 - self.momentum) * g_w;
                            *w_w -= self.learning_rate * (*v_w);
                        }
                        *vel_bias = self.momentum * (*vel_bias) + (1.0 - self.momentum) * grad_bias;
                        *bias -= self.learning_rate * (*vel_bias);
                    }
                    _ => panic!("Unexpected velocity type for dense layer"),
                }
            }
            _ => panic!("Mismatched weight types"),
        }
    }
}
