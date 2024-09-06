import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Input
import json

# Define and compile a simple model with a flatten layer
def create_flatten_model():
    inputs = Input(shape=(5, 5, 1))  # Define a simple 5x5 input
    flatten_layer = Flatten()(inputs)  # Apply the flatten layer
    model = Model(inputs=inputs, outputs=flatten_layer)
    return model

# Function to save the flattened output to a JSON file
def save_flatten_output_to_json(output, filename='flatten_output.json'):
    # Flatten the output to a 1D array
    output_flattened = output.flatten().tolist()
    
    # Save as JSON
    with open(filename, 'w') as f:
        json.dump(output_flattened, f, indent=2)
    print(f'Flatten output saved to {filename}')

# Function to save the input image used in the test to a JSON file
def save_image_to_json(image, filename='test_image.json'):
    # Convert the image matrix to a list and save as JSON
    image_list = image.squeeze().tolist()  # Remove the extra dimension
    with open(filename, 'w') as f:
        json.dump(image_list, f, indent=2)
    print(f'Image saved to {filename}')

# Main function
def main():
    # Create the flatten model
    model = create_flatten_model()
    print(model.summary())
    
    # Create test input data (5x5 image)
    test_input = np.array([[
        [ [1], [-2], [3], [-4], [5] ],
        [ [-6], [7], [-8], [9], [-10] ],
        [ [11], [-12], [13], [-14], [15] ],
        [ [-16], [17], [-18], [19], [-20] ],
        [ [21], [-22], [23], [-24], [25] ]
    ]], dtype=np.float32)  # Shape: (1, 5, 5, 1) for batch_size=1, 5x5 image, 1 channel
    
    # Save the input image to a JSON file
    save_image_to_json(test_input)
    
    # Get the flattened output of the model for the test input
    output = model.predict(test_input)
    
    # Save the flattened output to a JSON file
    save_flatten_output_to_json(output)

if __name__ == "__main__":
    main()
