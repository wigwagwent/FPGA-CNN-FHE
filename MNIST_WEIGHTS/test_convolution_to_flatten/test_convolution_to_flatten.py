import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import json

from tensorflow.keras.layers import Input

def create_cnn_model():
    inputs = Input(shape=(28, 28, 1))  # Define the input layer
    x = Conv2D(8, (3, 3), activation='relu')(inputs)
    x = Flatten()(x)
    outputs = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)  # Explicitly define the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# Function to preprocess MNIST data
def preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Define and compile the CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_weights_json(model, filename='weights.json'):
    with open(filename, 'r') as f:
        weights_data = json.load(f)

    data_index = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            conv_weights = []
            conv_biases = []
            filters = layer.filters
            input_channels = model.input_shape[-1]  # Use model input shape to get input channels
            print(f"Expected Conv2D shape: (3, 3, {input_channels}, {filters})")

            for i in range(filters):
                layer_data = weights_data[data_index]
                if layer_data['type'] != 'Convolution':
                    raise ValueError(f"Expected Convolution layer, got {layer_data['type']}")

                kernel = np.array(layer_data['kernel'])
                print(f"Kernel shape before reshape: {kernel.shape}")
                conv_weights.append(kernel)
                conv_biases.append(layer_data['bias'])
                data_index += 1

            # Stack and reshape kernel to (height, width, input_channels, output_channels)
            conv_weights = np.stack(conv_weights, axis=-1)
            conv_weights = np.expand_dims(conv_weights, axis=-2)
            conv_weights = conv_weights.reshape(3, 3, input_channels, filters)
            conv_biases = np.array(conv_biases)

            print(f"Conv2D layer shape: {layer.get_weights()[0].shape}")
            print(f"Loaded conv_weights shape: {conv_weights.shape}")
            print(f"Loaded conv_bias shape: {conv_biases.shape}")

            layer.set_weights([conv_weights, conv_biases])

        elif isinstance(layer, tf.keras.layers.Dense):
            dense_weights = []
            dense_biases = []
            units = layer.units
            input_features = layer.input_shape[1]  # Use the input_shape attribute of the Dense layer
            print(f"Expected Dense shape: ({input_features}, {units})")

            for i in range(units):
                layer_data = weights_data[data_index]
                if layer_data['type'] != 'Dense':
                    raise ValueError(f"Expected Dense layer, got {layer_data['type']}")

                weights = np.array(layer_data['weights'])
                print(f"Weights shape before reshape: {weights.shape}")
                dense_weights.append(weights)
                dense_biases.append(layer_data['bias'])
                data_index += 1

            # Stack and reshape weights to (input_features, output_features)
            dense_weights = np.stack(dense_weights, axis=-1)
            dense_biases = np.array(dense_biases)

            print(f"Dense layer shape: {layer.get_weights()[0].shape}")
            print(f"Loaded dense_weights shape: {dense_weights.shape}")
            print(f"Loaded dense_bias shape: {dense_biases.shape}")

            layer.set_weights([dense_weights, dense_biases])

    print("Weights loaded successfully.")


def main():
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    model = create_cnn_model()
    print(model.summary())

    # Check if weights file exists
    weights_file = 'weights.json'
    load_weights_json(model, weights_file)


    # Evaluate the model
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')


if __name__ == "__main__":
    main()
