pub mod layers;
mod activation;

pub type Quantized = f32;
pub type VecD1 = Vec<Quantized>;  // 1D vector for dense layer
pub type VecD2 = Vec<Vec<Quantized>>; // 2D vector for convolutional layer
pub type Kernel = [[Quantized; 3]; 3]; // 3x3 kernel

pub trait KernelExt {
    fn kernel_size() -> usize;
}

impl KernelExt for Kernel {
    fn kernel_size() -> usize {
        3 * 3
    }
}

#[derive(Clone)]
pub enum Weights {
    Convolution(Kernel, Quantized), // Convolutional layer weights and bias
    Dense(VecD1, Quantized), // Dense layer weights and bias
}

#[derive(Clone, Copy)]
pub enum Activation {
    ReLU(Quantized),
    Softmax,
}