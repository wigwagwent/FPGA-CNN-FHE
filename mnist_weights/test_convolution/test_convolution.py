import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input
import json

# Function to save model weights in JSON format
def save_weights_json(model, filename='test_weights.json'):
    weights_list = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            # Convolutional layer
            w, b = layer.get_weights()
            for i in range(w.shape[3]):  # For each output channel
                kernel = w[:,:,0,i].tolist()  # Take only the first channel for 3x3 kernel
                weights_list.append({
                    "type": "Convolution",
                    "kernel": kernel,
                    "bias": float(b[i])  # Save the bias for each output channel
                })

    # Write the weights to a JSON file
    with open(filename, 'w') as f:
        json.dump(weights_list, f, indent=2)
    print(f'Weights saved to {filename}')

# Function to preprocess MNIST data
def preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    return x_train, x_test

# Define and compile a simple CNN model with 1 convolutional layer and bias
def create_simple_cnn_model():
    inputs = Input(shape=(28, 28, 1))  # Define the input layer
    conv_layer = Conv2D(1, (3, 3), activation='linear', use_bias=True)(inputs)  # 1 filter (neuron) with bias
    
    # Create the model
    model = Model(inputs=inputs, outputs=conv_layer)
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Function to manually set the bias to a non-zero value
def set_bias_to_nonzero(model, bias_value=1.0):
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            weights = layer.get_weights()
            weights[1] = np.array([bias_value])  # Set bias to a specific non-zero value
            layer.set_weights(weights)

# Function to save the 2D matrix (output of the model) to a JSON file
def save_output_to_json(output, filename='output.json'):
    # Convert the output matrix to a list and save as JSON
    output_list = output.tolist()
    with open(filename, 'w') as f:
        json.dump(output_list, f, indent=2)
    print(f'Output saved to {filename}')

# Function to save the input image used in the test to a JSON file
def save_image_to_json(image, filename='test_image.json'):
    # Convert the image matrix to a list and save as JSON
    image_list = image.squeeze().tolist()  # Remove the extra dimension
    with open(filename, 'w') as f:
        json.dump(image_list, f, indent=2)
    print(f'Image saved to {filename}')

# Main function
def main():
    # Preprocess data
    x_train, x_test = preprocess_data()
    
    # Create the simple CNN model
    model = create_simple_cnn_model()
    print(model.summary())
    
    # Manually set the bias to a non-zero value (e.g., 1.0)
    set_bias_to_nonzero(model, bias_value=1.0)
    
    # Use a sample from the test data (first image) for testing
    sample_input = x_test[0:1]
    
    # Save the input image to a JSON file
    save_image_to_json(sample_input)
    
    # Get the output of the model for the sample input
    output = model.predict(sample_input)
    
    # The output is a 4D tensor with shape (1, height, width, 1)
    # We need to remove the batch and channel dimensions to get a 2D matrix
    output_2d = np.squeeze(output, axis=(0, 3))  # Remove batch and channel dimensions
    
    # Save the 2D output to a JSON file
    save_output_to_json(output_2d)
    
    # Save the model weights (including the bias) to a JSON file
    save_weights_json(model)

if __name__ == "__main__":
    main()
