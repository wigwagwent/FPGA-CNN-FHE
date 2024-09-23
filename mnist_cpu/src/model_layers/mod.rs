use fhe_ckks::{DoubleSized, Plaintext};
use num_traits::{PrimInt, Signed};
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
pub enum WeightsFhe<T, const N: usize>
where
    T: PrimInt + Signed + DoubleSized,
{
    Convolution {
        kernel: VecD2,
        bias: Quantized,
    },
    Dense {
        weights: Plaintext<T, N>,
        bias: Plaintext<T, N>,
    },
}

impl<T, const N: usize> From<Weights> for WeightsFhe<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    fn from(weights: Weights) -> Self {
        match weights {
            Weights::Convolution { kernel, bias } => WeightsFhe::Convolution { kernel, bias },
            Weights::Dense { weights, bias } => WeightsFhe::Dense {
                weights: Plaintext::from_f32(weights.clone(), 15),
                bias: Plaintext::from_f32(vec![bias; weights.len()], 15),
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
    pub fn new(learning_rate: Quantized, momentum: Quantized) -> Self {
        SGDOptimizer {
            learning_rate,
            momentum,
            dense_velocities: Vec::new(),
            conv_velocities: Vec::new(),
        }
    }

    pub fn adjust_learning_rate(&mut self, epoch: usize, factor: Quantized, step_size: usize) {
        if epoch > 0 && epoch % step_size == 0 {
            self.learning_rate *= factor;
            println!("Adjusted learning rate to: {}", self.learning_rate);
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
                Weights::Dense { weights, .. } => Weights::Dense {
                    weights: vec![0.0; weights.len()],
                    bias: 0.0,
                },
                _ => panic!("Unexpected weight type for dense layer"),
            })
            .collect();

        self.conv_velocities = conv_weights
            .iter()
            .map(|w| match w {
                Weights::Convolution { kernel, .. } => Weights::Convolution {
                    kernel: vec![vec![0.0; kernel[0].len()]; kernel.len()],
                    bias: 0.0,
                },
                _ => panic!("Unexpected weight type for convolution layer"),
            })
            .collect();
    }

    // Update the weights using momentum and the gradients for both dense and convolutional layers
    pub fn update_all(
        &mut self,
        dense_weights: &mut Vec<Weights>,
        dense_gradients: &Vec<Weights>,
        conv_weights: &mut Vec<Weights>,
        conv_gradients: &Vec<Weights>,
    ) {
        // Update dense weights
        for ((weight, gradient), velocity) in dense_weights
            .iter_mut()
            .zip(dense_gradients.iter())
            .zip(self.dense_velocities.iter_mut())
        {
            match (weight, gradient, velocity) {
                (
                    Weights::Dense { weights: w, bias: b },
                    Weights::Dense {
                        weights: grad_weights,
                        bias: grad_bias,
                    },
                    Weights::Dense {
                        weights: vel_weights,
                        bias: vel_bias,
                    },
                ) => {
                    for (w, (v, g)) in w.iter_mut().zip(vel_weights.iter_mut().zip(grad_weights)) {
                        *v = self.momentum * (*v) + (1.0 - self.momentum) * g;
                        *w -= self.learning_rate * (*v);
                    }
                    *vel_bias = self.momentum * (*vel_bias) + (1.0 - self.momentum) * grad_bias;
                    *b -= self.learning_rate * (*vel_bias);
                }
                _ => panic!("Mismatched weight types for dense layers."),
            }
        }

        // Update convolutional weights
        for ((weight, gradient), velocity) in conv_weights
            .iter_mut()
            .zip(conv_gradients.iter())
            .zip(self.conv_velocities.iter_mut())
        {
            match (weight, gradient, velocity) {
                (
                    Weights::Convolution { kernel, bias },
                    Weights::Convolution {
                        kernel: grad_kernel,
                        bias: grad_bias,
                    },
                    Weights::Convolution {
                        kernel: vel_kernel,
                        bias: vel_bias,
                    },
                ) => {
                    for ((k, v), g) in kernel
                        .iter_mut()
                        .zip(vel_kernel.iter_mut())
                        .zip(grad_kernel.iter())
                    {
                        for (k, (v, g)) in k.iter_mut().zip(v.iter_mut().zip(g)) {
                            *v = self.momentum * (*v) + (1.0 - self.momentum) * g;
                            *k -= self.learning_rate * (*v);
                        }
                    }
                    *vel_bias = self.momentum * (*vel_bias) + (1.0 - self.momentum) * grad_bias;
                    *bias -= self.learning_rate * (*vel_bias);
                }
                _ => panic!("Mismatched weight types for convolution layers."),
            }
        }
    }
}
