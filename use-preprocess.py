import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from numpy.linalg import pinv

# Load preprocessed data from CSV files
X_train = pd.read_csv('processed_X_train.csv')
X_test = pd.read_csv('processed_X_test.csv')
y_train = pd.read_csv('processed_y_train.csv').values.ravel()
y_test = pd.read_csv('processed_y_test.csv').values.ravel()

# Identify and keep only common columns across all datasets
common_columns = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_columns]
X_test = X_test[common_columns]

# Check initial shapes after aligning columns
print("Initial shapes after aligning columns:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# # Apply PCA with a fixed number of components within the limit
n_components = min(15, X_train.shape[1])

# Check shapes after PCA to ensure consistency
print("\nShapes after PCA:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

# Define the ELM model class
class ELMClassifier:
    def __init__(self, input_size, hidden_size, activation='relu'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        # Random weights and biases for hidden layer
        self.input_weights = np.random.randn(input_size, hidden_size)
        self.bias = np.random.randn(hidden_size)
    
    def _activation_function(self, X):
        if self.activation == 'relu':
            return np.maximum(0, X)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-X))
        else:
            raise ValueError("Unsupported activation function.")
    
    def fit(self, X, y):
        # Apply input weights and bias, then activation function
        H = self._activation_function(np.dot(X, self.input_weights) + self.bias)
        # Calculate output weights (beta) using the Moore-Penrose pseudoinverse
        self.output_weights = np.dot(pinv(H), y)
    
    def predict(self, X):
        # Apply input weights and bias, then activation function
        H = self._activation_function(np.dot(X, self.input_weights) + self.bias)
        # Multiply by output weights to get predictions
        return np.dot(H, self.output_weights)

# Binarize labels for multi-class classification
lb = LabelBinarizer()
y_train_bin = lb.fit_transform(y_train)
y_test_bin = lb.transform(y_test)

# Check the shape of binarized labels
print("\nShapes of binarized labels:")
print("y_train_bin shape:", y_train_bin.shape)
print("y_test_bin shape:", y_test_bin.shape)

# Instantiate and train the ELM model
elm = ELMClassifier(input_size=n_components, hidden_size=100, activation='relu')
elm.fit(X_train, y_train_bin)

# Predict on validation and test sets
y_test_pred = elm.predict(X_test)

# Convert predictions back to label format
y_test_pred = lb.inverse_transform(y_test_pred)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))
