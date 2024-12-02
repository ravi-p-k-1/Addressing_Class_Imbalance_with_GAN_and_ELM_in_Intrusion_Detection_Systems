import os
import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelBinarizer
from numpy.linalg import pinv

# Load preprocessed data
X_train = pd.read_csv('processed_X_train.csv')
X_test = pd.read_csv('real_processed_X_test.csv')
y_train = pd.read_csv('processed_y_train.csv').values.ravel()
y_test = pd.read_csv('real_processed_y_test.csv').values.ravel()

# Identify and keep only common columns
common_columns = X_train.columns.intersection(X_test.columns)
X_train = X_train[common_columns]
X_test = X_test[common_columns]

# Apply PCA or define components
n_components = min(15, X_train.shape[1])
print(f"Number of PCA components: {n_components}")

# Binarize labels for multi-class classification
lb = LabelBinarizer()
y_train_bin = lb.fit_transform(y_train)
y_test_bin = lb.transform(y_test)

print("\nShapes of binarized labels:")
print(f"y_train_bin shape: {y_train_bin.shape}")
print(f"y_test_bin shape: {y_test_bin.shape}")

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
        print(f"Fitting model: X shape = {X.shape}, y shape = {y.shape}")
        # Apply input weights and bias, then activation function
        H = self._activation_function(np.dot(X, self.input_weights) + self.bias)
        print(f"H shape: {H.shape}")

        # Ensure label shape matches the number of samples
        if y.ndim == 1:
            y = y[:, np.newaxis]

        if H.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch: H rows {H.shape[0]} != y rows {y.shape[0]}")

        # Calculate output weights using the Moore-Penrose pseudoinverse
        self.output_weights = np.dot(pinv(H), y)

    def predict(self, X):
        H = self._activation_function(np.dot(X, self.input_weights) + self.bias)
        predictions = np.dot(H, self.output_weights)
        return predictions

# Load or train the model
if os.path.exists('elm_model.pkl') and input("Load saved model? (y/n): ").lower() == 'y':
    elm = joblib.load('elm_model.pkl')
    print("Loaded saved model.")
else:
    print("No saved model found or opted to train a new model.")
    elm = ELMClassifier(input_size=X_train.shape[1], hidden_size=100, activation='relu')
    elm.fit(X_train, y_train_bin)

# Predict on test set
y_test_pred = elm.predict(X_test)

# Convert predictions back to label format
y_test_pred = lb.inverse_transform(y_test_pred)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2f}")
print("Test Classification Report:")
print(classification_report(y_test, y_test_pred))


if input('Save model or discard? y to save').rstrip().lower() == 'y':
    joblib.dump(elm, 'elm_model.pkl')
    print("Model trained and saved.")
else:
    print('Not saved!')
