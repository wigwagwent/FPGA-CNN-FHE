import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Softmax
import json

# Function to save data to JSON files
def save_to_json(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to", filename)

# Define and compile a simple model with a softmax activation
def create_softmax_model(input_shape):
    inputs = Input(shape=input_shape)
    dense_layer = Dense(1, use_bias=True)(inputs)  # Dense layer
    softmax_layer = Softmax()(dense_layer)  # Softmax layer after dense
    model = Model(inputs=inputs, outputs=softmax_layer)
    return model

# Main function to create test cases and generate JSON files
def main():
    # Create the softmax model
    model = create_softmax_model((5,))
    
    # Create test input data (matching the output from flatten layer)
    flattened_input = np.array([1.0, -2.0, 3.0, -4.0, 5.0], dtype=np.float32)

    # Set custom weights and bias for the dense layer
    for layer in model.layers:
        if isinstance(layer, Dense):
            weights = np.array([[0.5], [1.0], [-0.5], [0.75], [-1.0]])
            bias = np.array([0.25])
            layer.set_weights([weights, bias])

    # Save the flattened input to a JSON file
    save_to_json(flattened_input.tolist(), 'flatten_input.json')

    # Get the softmax output of the model for the flattened input
    softmax_output = model.predict(flattened_input.reshape(1, -1)).flatten()

    # Save the softmax output to a JSON file
    save_to_json(softmax_output.tolist(), 'softmax_output.json')

    # Save the weights to a JSON file
    weights, bias = model.layers[1].get_weights()
    weights_json = {
        "weights": weights.flatten().tolist(),
        "bias": bias.flatten().tolist()[0]  # Single bias for the output neuron
    }
    save_to_json([weights_json], 'test_weights.json')

if __name__ == "__main__":
    main()
