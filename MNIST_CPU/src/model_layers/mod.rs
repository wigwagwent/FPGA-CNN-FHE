pub mod layers;
mod activation;

pub type Quantized = f32;
pub type VecD1 = Vec<Quantized>;  // 1D vector for dense layer
pub type VecD2 = Vec<Vec<Quantized>>; // 2D vector for convolutional layer
pub type Kernel = [[Quantized; 3]; 3]; // 3x3 kernel

pub trait KernelExt {
    fn kernel_size() -> usize;
    fn new(weights: Vec<Vec<Vec<Quantized>>>) -> Vec<Kernel>; // Now returns Vec<Kernel> to handle multiple filters
}

impl KernelExt for Kernel {
    fn kernel_size() -> usize {
        3 * 3
    }

    // Implementation of Kernel::new to create a vector of 3x3 kernels from a 3D vector
    fn new(weights: Vec<Vec<Vec<Quantized>>>) -> Vec<Kernel> {
        let mut kernels: Vec<Kernel> = Vec::new();

        // Ensure that the input data has the correct format
        for weight in weights {
            assert_eq!(weight.len(), 3, "Expected 3 rows for each 3x3 kernel");
            for row in &weight {
                assert_eq!(row.len(), 3, "Expected 3 columns for each 3x3 kernel");
            }

            // Convert Vec<Vec<Quantized>> to [[Quantized; 3]; 3]
            let mut kernel: Kernel = [[0.0; 3]; 3];
            for i in 0..3 {
                for j in 0..3 {
                    kernel[i][j] = weight[i][j];
                }
            }
            kernels.push(kernel);
        }

        kernels
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