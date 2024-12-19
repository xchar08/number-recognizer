import numpy as np
from matplotlib import pyplot as plt
import random

# Motivational messages
MESSAGES = [
    "Do a barrel roll!",
    "Pretty Soary, ey?",
    "Brought to you by the MIT Mostec trio!",
    "Let's move onto another one, that was too easy!"
]

# Load MNIST dataset
def load_mnist(file_path='mnist.npz'):
    with np.load(file_path) as data:
        images, labels = data["x_train"], data["y_train"]
    images = images.astype("float32") / 255  # Normalize pixel values
    images = images.reshape(images.shape[0], -1)  # Flatten images
    labels = np.eye(10)[labels]  # One-hot encode labels
    return images, labels

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Sigmoid derivative for backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize network parameters
def initialize_network(input_size, hidden_size, output_size):
    weights = {
        "input_hidden": np.random.uniform(-0.5, 0.5, (hidden_size, input_size)),
        "hidden_output": np.random.uniform(-0.5, 0.5, (output_size, hidden_size))
    }
    biases = {
        "input_hidden": np.zeros((hidden_size, 1)),
        "hidden_output": np.zeros((output_size, 1))
    }
    return weights, biases

# Forward propagation
def forward_propagation(inputs, weights, biases):
    hidden_pre = biases["input_hidden"] + weights["input_hidden"] @ inputs
    hidden = sigmoid(hidden_pre)

    output_pre = biases["hidden_output"] + weights["hidden_output"] @ hidden
    output = sigmoid(output_pre)

    return hidden, output

# Backward propagation
def backward_propagation(inputs, hidden, output, label, weights, biases, learning_rate):
    # Output layer error and update
    output_error = output - label
    weights["hidden_output"] -= learning_rate * output_error @ hidden.T
    biases["hidden_output"] -= learning_rate * output_error

    # Hidden layer error and update
    hidden_error = (weights["hidden_output"].T @ output_error) * sigmoid_derivative(hidden)
    weights["input_hidden"] -= learning_rate * hidden_error @ inputs.T
    biases["input_hidden"] -= learning_rate * hidden_error

def train(images, labels, weights, biases, learning_rate, epochs):
    for epoch in range(epochs):
        correct_evaluations = 0

        for img, lbl in zip(images, labels):
            img = img.reshape(-1, 1)  # Reshape to column vector
            lbl = lbl.reshape(-1, 1)  # Reshape to column vector

            # Forward pass
            hidden, output = forward_propagation(img, weights, biases)

            # Accuracy calculation
            if np.argmax(output) == np.argmax(lbl):
                correct_evaluations += 1

            # Backward pass
            backward_propagation(img, hidden, output, lbl, weights, biases, learning_rate)

        accuracy = round((correct_evaluations / len(images)) * 100, 2)
        print(f"Epoch {epoch + 1}/{epochs} - Accuracy: {accuracy}%")

# Predict and display result
def predict_and_display(images, weights, biases):
    while True:
        try:
            input_index = int(input("Enter a number between 1 and 60000: ")) - 1
            img = images[input_index].reshape(28, 28)

            plt.imshow(img, cmap="Greys")

            img = img.reshape(-1, 1)  # Flatten and reshape to column vector
            _, output = forward_propagation(img, weights, biases)

            prediction = np.argmax(output)
            plt.title(f"It is a {prediction}. " + random.choice(MESSAGES))
            plt.show()
        except (ValueError, IndexError):
            print("Invalid input. Please try again.")

# Main execution
if __name__ == "__main__":
    images, labels = load_mnist()
    weights, biases = initialize_network(input_size=784, hidden_size=20, output_size=10)

    train(images, labels, weights, biases, learning_rate=0.01, epochs=10)
    predict_and_display(images, weights, biases)
