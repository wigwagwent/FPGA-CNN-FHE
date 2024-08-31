import numpy as np
import torch
from torchvision import datasets, transforms
from Pyfhel import Pyfhel, PyCtxt
import torch.nn.functional as F

class CPU:
    def __init__(self, fhe):
        """
        Initialize the CPU with a Pyfhel object for encryption and decryption.
        """
        self.fhe = fhe
        self.load_data()
    
    def load_data(self):
        """
        Load the MNIST dataset and select a single image for processing.
        """
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
        self.image, self.label = test_dataset[0]  # Take the first image
        self.image = self.image.numpy().flatten()  # Flatten the image to 1D array
        self.image = self.image * 255  # Scale pixel values to [0, 255]
        self.image = self.image.astype(np.float32)
        print(f"Loaded image with label: {self.label}")
    
    def encrypt_data(self):
        """
        Encrypt the image data using FHE.
        """
        encrypted_data = [self.fhe.encryptFrac(x) for x in self.image]
        print("Data encrypted.")
        return encrypted_data
    
    def decrypt_result(self, encrypted_result):
        """
        Decrypt the result data using FHE.
        """
        decrypted_data = [self.fhe.decryptFrac(ctxt) for ctxt in encrypted_result]
        decrypted_data = np.array(decrypted_data)
        print("Result decrypted.")
        return decrypted_data

class FPGA:
    def __init__(self, fhe):
        """
        Initialize the FPGA with a Pyfhel object for performing homomorphic operations.
        """
        self.fhe = fhe
    
    def perform_cnn_operations(self, encrypted_data):
        """
        Simulate CNN operations on encrypted data.
        For simplicity, we'll perform a linear transformation followed by a non-linear activation.
        """
        # Simulate a simple linear layer (e.g., a dot product with weights)
        weights = np.random.randn(len(encrypted_data)).astype(np.float32)
        encrypted_weights = [self.fhe.encryptFrac(w) for w in weights]
        
        # Homomorphic multiplication (element-wise)
        multiplied = [ctxt * w_ctxt for ctxt, w_ctxt in zip(encrypted_data, encrypted_weights)]
        
        # Homomorphic addition (sum the products)
        summed = multiplied[0]
        for ctxt in multiplied[1:]:
            summed += ctxt
        
        # Simulate ReLU activation (using squared value as a placeholder)
        activated = summed * summed  # Note: Actual ReLU is non-linear and not directly supported
        
        print("CNN operations performed on FPGA.")
        return [activated]  # Returning as a list to maintain consistency

def main():
    # Initialize Pyfhel for FHE operations
    fhe = Pyfhel()
    fhe.contextGen(p=65537, m=8192, sec=128)  # Parameters can be adjusted
    fhe.keyGen()
    print("FHE context and keys generated.")
    
    # Initialize CPU and FPGA
    cpu = CPU(fhe)
    fpga = FPGA(fhe)
    
    # CPU encrypts the data
    encrypted_data = cpu.encrypt_data()
    
    # Pass encrypted data to FPGA for CNN operations
    encrypted_result = fpga.perform_cnn_operations(encrypted_data)
    
    # Pass encrypted result back to CPU for decryption
    decrypted_result = cpu.decrypt_result(encrypted_result)
    
    # For demonstration, print the decrypted result
    print(f"Decrypted result (first 10 values): {decrypted_result[:10]}")

if __name__ == "__main__":
    main()
