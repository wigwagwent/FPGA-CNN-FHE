import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

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
def quantize_weights(weights, bit_width=16):
    scale = 2 ** (bit_width - 1) - 1
    quantized_weights = [np.clip(np.round(w * scale), -scale, scale).astype(np.int16) for w in weights]
    return quantized_weights

# Function to save weights in binary format
def save_weights_bin(weights, filename='weights.bin'):
    with open(filename, 'wb') as f:
        for w in weights:
            # Flatten the weights and write as binary data
            f.write(w.tobytes())

# Define and compile the CNN model
def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
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
    trained_model = train_and_evaluate_model(model, x_train, y_train, x_test, y_test)
    weights = trained_model.get_weights()
    quantized_weights = quantize_weights(weights)
    save_weights_bin(quantized_weights)

if __name__ == "__main__":
    main()