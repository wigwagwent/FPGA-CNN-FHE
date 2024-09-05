import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import json

# Function to preprocess MNIST data
def preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.reshape((-1, 28, 28, 1))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)

# Function to quantize weights
def quantize_weights(weights, bit_width=8):
    scale = 2 ** bit_width - 1
    quantized_weights = [np.clip(np.round(w * scale), 0, scale).astype(np.uint8) for w in weights]
    return quantized_weights

# Function to save weights in JSON format
def save_weights_json(model, filename='weights.json'):
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
                    "bias": float(b[i])
                })

        elif isinstance(layer, tf.keras.layers.Dense):
            # Dense layer
            w, b = layer.get_weights()
            for i in range(w.shape[1]):  # For each neuron
                weights_list.append({
                    "type": "Dense",
                    "weights": w[:, i].tolist(),
                    "bias": float(b[i])
                })

    # Write the weights to a JSON file
    with open(filename, 'w') as f:
        json.dump(weights_list, f, indent=2)

# Define and compile the CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train and evaluate the CNN model
def train_and_evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc:.4f}')
    return model

# Main function
def main():
    (x_train, y_train), (x_test, y_test) = preprocess_data()
    model = create_cnn_model()
    print(model.summary())
    trained_model = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)
    save_weights_json(trained_model)

if __name__ == "__main__":
    main()
