# FPGA-CNN-FHE
Accelerating Privacy-Preserving Convolutional Neural Networks on FPGAs Using Fully Homomorphic Encryption

# Project Overview

This project focuses on training an MNIST model in Python and transferring the trained weights to an FPGA for Fully Homomorphic Encryption (FHE) computations. The goal is to enable secure and privacy-preserving machine learning inference on encrypted data. Different models could follow a similar approch to be transfered over to FPGA for computational accelerations.

## Key Features

- Train an MNIST model using Python.
- Implement Fully Homomorphic Encryption (FHE) on an FPGA.
- Transfer the trained model weights to the FPGA for encrypted inference.
- Evaluate the accuracy and performance of the FHE-based inference.

## Project Structure

The project is organized into the following directories:

- `ModelGeneration`: Contains the Python code for training the MNIST model.
- `FPGAImplementation`: Includes the FPGA implementation code for FHE computations.
- `Evaluation`: Contains scripts and notebooks for evaluating the accuracy and performance of the FHE-based inference.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository.
2. Navigate to the `ModelGeneration` directory.
3. Run the Python script to train the MNIST model.
4. Transfer the trained weights to the FPGA using the provided FPGA implementation code.
5. Evaluate the accuracy and performance of the FHE-based inference using the scripts and notebooks in the `Evaluation` directory.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- TensorFlow
- PyTorch
- FPGA development tools (specifics depend on the FPGA platform being used)

## License

This project is dual licensed under the MIT and APACHE 2.0 Licenses. See the [MIT-LICENSE](./MIT-LICENSE) or [APACHE-LICENSE](./APACHE-LICENSE) file for more details.