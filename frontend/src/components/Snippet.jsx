import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';

const codeString = `import numpy as np
import pickle
import time

# Define a simple Deep Neural Network (DNN) class
class DNN:
    def __init__(self, sizes=[784, 128, 64, 10], epochs=10, lr=0.001):
        """
        Initializes the neural network with:
        - sizes: A list indicating the number of neurons in each layer 
        - epochs: Number of passes over the entire training dataset.
        - lr: Learning rate used for gradient descent updates.
        """
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

        # Assign sizes for clarity
        input_layer = sizes[0]
        hidden_1 = sizes[1]
        hidden_2 = sizes[2]
        output_layer = sizes[3]

        # Initialize weights using a Gaussian distribution scaled by the size of the layer 
        # (He-like initialization)
        self.params = {
            # weights from input to hidden1
            'W1': np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1), 
            # weights from hidden1 to hidden2
            'W2': np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),    
            # weights from hidden2 to output
            'W3': np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer) 
        }

    def sigmoid(self, x, derivative=False):
        """
        Sigmoid activation function used in hidden layers.
        If derivative=True, returns the derivative for use in backpropagation.
        """
        if derivative:
            return (np.exp(-x)) / ((np.exp(-x) + 1) ** 2)  # Derivative of sigmoid
        return 1 / (1 + np.exp(-x))  # Sigmoid function

    def softmax(self, x, derivative=False):
        """
        Softmax activation function used for the output layer 
        to convert raw scores to probabilities.
        If derivative=True, returns the simplified gradient assuming cross-entropy loss is used.
        """
        exps = np.exp(x - x.max())  # Subtracting max for numerical stability
        if derivative:
            s = exps / np.sum(exps, axis=0)
            return s * (1 - s)  # Element-wise gradient approximation
        return exps / np.sum(exps, axis=0)

    def forward_pass(self, x_train):
        """
        Perform a forward pass through the network, computing activations at each layer.
        Stores intermediate values in self.params for use in backpropagation.
        """
        params = self.params
        params['A0'] = x_train  # Input layer activation (just the input)

        # First hidden layer
        params['Z1'] = np.dot(params['W1'], params['A0'])  # Linear transformation
        params['A1'] = self.sigmoid(params['Z1'])          # Activation

        # Second hidden layer
        params['Z2'] = np.dot(params['W2'], params['A1'])
        params['A2'] = self.sigmoid(params['Z2'])

        # Output layer
        params['Z3'] = np.dot(params['W3'], params['A2'])
        params['A3'] = self.softmax(params['Z3'])  # Probabilities

        return params['A3']  # Return final prediction

    def update_weights(self, change_w):
        """
        Update the network's weights using the gradients computed during backpropagation.
        """
        for key, val in change_w.items():
            self.params[key] -= self.lr * val  # Gradient descent step

    def backward_pass(self, y_train, output):
        """
        Perform backpropagation to compute gradients of the loss with respect to weights.
        Returns a dictionary of weight gradients.
        """
        params = self.params
        change_w = {}

        # Error at output layer (cross-entropy loss with softmax)
        error = output - y_train
        change_w['W3'] = np.outer(error, params['A2'])  # Gradient for W3

        # Propagate error to hidden layer 2
        error = np.dot(params['W3'].T, error) * self.sigmoid(params['Z2'], derivative=True)
        change_w['W2'] = np.outer(error, params['A1'])  # Gradient for W2

        # Propagate error to hidden layer 1
        error = np.dot(params['W2'].T, error) * self.sigmoid(params['Z1'], derivative=True)
        change_w['W1'] = np.outer(error, params['A0'])  # Gradient for W1

        return change_w

    def compute_accuracy(self, test_data):
        """
        Evaluate the model's accuracy on the test dataset.
        Each example is classified, and the prediction is compared to the true label.
        """
        predictions = []
        for x in test_data:
            values = x.split(",")
            inputs = (np.asarray(values[1:], dtype=float) / 255.0)  # Normalize pixel values
            targets = np.zeros(10) + 0.01  # Avoid zero outputs (for numerical stability)
            targets[int(values[0])] = 0.99  # One-hot encoding with high confidence

            output = self.forward_pass(inputs)
            pred = np.argmax(output)  # Predicted class
            predictions.append(pred == np.argmax(targets))  # Check if correct

        return np.mean(predictions)  # Fraction of correct predictions

    def train(self, train_list, test_list):
        """
        Train the model on the provided training data for the given number of epochs.
        After each epoch, the model’s accuracy is evaluated on the test set.
        """
        start_time = time.time()
        for i in range(self.epochs):
            for x in train_list:
                values = x.split(",")
                inputs = (np.asarray(values[1:], dtype=float) / 255.0)  # Normalize inputs
                targets = np.zeros(10) + 0.01
                targets[int(values[0])] = 0.99  # One-hot encoding

                output = self.forward_pass(inputs)  # Forward pass
                change_w = self.backward_pass(targets, output)  # Backpropagation
                self.update_weights(change_w)  # Update weights

            accuracy = self.compute_accuracy(test_list)  # Evaluate model
            print(f"Epoch {i+1}/{self.epochs} - Accuracy: {accuracy:.4f}")

def save_model_weights(model, filename='mnist_model_weights.pkl'):
    """
    Saves the trained weights of the model to a file using pickle.
    This allows the model to be reloaded later without retraining.
    """
    weights = {
        'W1': model.params['W1'],
        'W2': model.params['W2'],
        'W3': model.params['W3']
    }

    with open(filename, 'wb') as f:
        pickle.dump(weights, f)

# Example usage
dnn = DNN(sizes=[784, 128, 64, 10], epochs=10, lr=0.001)
dnn.train(train_list, test_list)
save_model_weights(dnn)
`;

export default function CodeSnippet() {
  return (
    <div className="scroll-container" style={{
        maxHeight: '800px',
        overflowY: 'auto',
        border: '2px solid rgb(19, 0, 233)',
        borderRadius: '8px',
        boxShadow: '0 0 15px rgb(19, 0, 233)',
        margin: '2rem auto',
        maxWidth: '900px',
  }}>
      <SyntaxHighlighter
        language="python"
        style={oneDark}
        customStyle={{
          margin: 0,
          padding: '1rem',
          backgroundColor: '#282c34',
          borderRadius: '8px',
        }}
      >
        {codeString}
      </SyntaxHighlighter>
    </div>
  );
}