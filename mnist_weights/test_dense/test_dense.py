import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import json

# Function to save data to JSON files
def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to", filename)

# Define and compile a simple model with a dense layer
def create_dense_model(input_shape):
    inputs = Input(shape=input_shape)  # Define input shape
    dense_layer = Dense(1, use_bias=True)(inputs)  # Apply a dense layer with 1 output neuron
    model = Model(inputs=inputs, outputs=dense_layer)
    return model

# Manually set weights for testing the dense layer
def set_test_weights(model):
    for layer in model.layers:
        if isinstance(layer, Dense):
            # Set the weights and bias for the dense layer
            weights = np.array([[0.5], [1.0], [-0.5], [0.75], [-1.0]])  # 5 inputs, 1 output
            bias = np.array([0.25])  # Single bias for the output neuron
            layer.set_weights([weights, bias])

# Main function to create test cases and generate JSON files
def main():
    # Create the dense model
    model = create_dense_model((5,))  # Input shape is 5 (matching flatten_input)
    
    # Set custom weights and bias for the dense layer
    set_test_weights(model)
    
    # Create flattened input data (matching output from the flatten layer)
    flattened_input = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)  # Shape (5,)
    
    # Save the flattened input to a JSON file
    save_to_json(flattened_input.tolist(), 'flatten_input.json')

    # Get the output of the dense layer for the flattened input
    dense_output = model.predict(flattened_input.reshape(1, -1)).flatten()
    
    # Save the dense layer output to a JSON file
    save_to_json(dense_output.tolist(), 'dense_output.json')

    # Save the dense layer weights to a JSON file (to be used in the Rust test)
    weights, bias = model.layers[1].get_weights()
    weights_json = {
        "weights": weights.flatten().tolist(),
        "bias": bias.flatten().tolist()[0]  # Bias is a single float
    }
    save_to_json([weights_json], 'test_weights.json')

if __name__ == "__main__":
    main()
