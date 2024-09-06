import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
import json

# Define and compile a simple CNN model with ReLU activation
def create_relu_cnn_model():
    inputs = Input(shape=(5, 5, 1))  # Define a simple 5x5 input
    conv_layer = Conv2D(1, (3, 3), activation='relu', use_bias=True)(inputs)  # ReLU activation with bias
    model = Model(inputs=inputs, outputs=conv_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Manually set weights for testing ReLU activation
def set_test_weights(model):
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            # Set weights with both negative and positive values
            weights = np.array([
                [[[-1]], [[2]], [[-3]]],
                [[[4]], [[-5]], [[6]]],
                [[[-7]], [[8]], [[-9]]]
            ])  # Shape (3, 3, 1, 1) for a 3x3 filter, 1 input channel, 1 output channel
            bias = np.array([1.0])  # Adding a bias of 1.0
            layer.set_weights([weights, bias])

# Function to save the model weights in JSON format
def save_weights_json(model, filename='test_weights.json'):
    weights_list = []

    for layer in model.layers:
        if isinstance(layer, Conv2D):
            # Get weights and bias
            w, b = layer.get_weights()
            kernel = w[:, :, 0, 0].tolist()  # Extract the first channel
            bias = b[0].tolist()

            weights_list.append({
                "type": "Convolution",
                "kernel": kernel,
                "bias": bias
            })

    # Write the weights to a JSON file
    with open(filename, 'w') as f:
        json.dump(weights_list, f, indent=2)
    print(f"Weights saved to {filename}")

# Function to save the input image used in the test to a JSON file
def save_image_to_json(image, filename='test_image.json'):
    image_list = image.squeeze().tolist()  # Remove the extra dimension
    with open(filename, 'w') as f:
        json.dump(image_list, f, indent=2)
    print(f"Image saved to {filename}")

# Function to save the output of the model to a JSON file
def save_output_to_json(output, filename='output.json'):
    output_list = output.squeeze().tolist()  # Remove the extra dimension
    with open(filename, 'w') as f:
        json.dump(output_list, f, indent=2)
    print(f"Output saved to {filename}")

# Main function to run the test and save files
def main():
    # Create a simple CNN model with ReLU activation
    model = create_relu_cnn_model()
    
    # Set custom test weights with both negative and positive values
    set_test_weights(model)
    
    # Create test input data (5x5 image with positive and negative values)
    test_input = np.array([[
        [ [1], [-2], [3], [-4], [5] ],
        [ [-6], [7], [-8], [9], [-10] ],
        [ [11], [-12], [13], [-14], [15] ],
        [ [-16], [17], [-18], [19], [-20] ],
        [ [21], [-22], [23], [-24], [25] ]
    ]], dtype=np.float32)  # Shape: (1, 5, 5, 1)
    
    # Save the input image to a JSON file
    save_image_to_json(test_input, 'test_image.json')
    
    # Get the output of the model for the sample input
    output = model.predict(test_input)
    
    # Save the 2D output to a JSON file
    save_output_to_json(output, 'output.json')
    
    # Save the model weights to a JSON file
    save_weights_json(model, 'test_weights.json')

if __name__ == "__main__":
    main()
