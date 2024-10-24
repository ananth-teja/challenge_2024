import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Generate sample data (1000 samples with 3 input features and 3 output features)
data = {
    'input1': np.random.rand(1000),
    'input2': np.random.rand(1000),
    'input3': np.random.rand(1000),
    'output1': np.random.rand(1000),
    'output2': np.random.rand(1000),
    'output3': np.random.rand(1000)
}

# Load data into a pandas DataFrame
df = pd.DataFrame(data)

# Separate inputs (features) and outputs (targets)
X = df[['input1', 'input2', 'input3']]  # Input features (3 columns)
y = df[['output1', 'output2', 'output3']]  # Output features (3 columns)

print(X.head())
print(y.head())

# Convert pandas DataFrames to numpy arrays for TensorFlow
X = X.values
y = y.values

# Create the neural network model
model = Sequential()

# Input layer (3 input neurons) and a hidden layer (10 neurons)
model.add(Dense(10, input_dim=3, activation='relu'))

# Another hidden layer with 8 neurons
model.add(Dense(8, activation='relu'))

# Output layer (3 neurons for 3 outputs)
model.add(Dense(3))

# Compile the model (using MSE for regression)
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# Sample prediction with a new input DataFrame
new_data = {'input1': [0.1], 'input2': [0.2], 'input3': [0.3]}
X_new = pd.DataFrame(new_data).values

# Predict using the trained model
y_pred = model.predict(X_new)

print("New input:", X_new)
print("Predicted output for new input:", y_pred)
