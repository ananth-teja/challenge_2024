import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Create a sample dataset of 10 rows, 3 input columns, and a single target value
X = pd.DataFrame({
    'Feature1': np.random.rand(10),
    'Feature2': np.random.rand(10),
    'Feature3': np.random.rand(10)
})
print(X)

# Define a single target value (e.g., average or some other computed outcome)
y = pd.Series([np.random.rand()])  # Target is just one value

# Randomly modify 1 row in the input data
random_index = np.random.choice(X.index, size=1, replace=False)
print("\nModified Row :")
print(random_index)
X.loc[random_index] += np.random.normal(0, 1, X.shape[1])  # Modify by adding noise

# Print the modified input data
print("Modified Input Data (10 rows):")
print(X)

# Convert the input data into a single feature vector (flattening the rows)
X_flat = X.values.flatten().reshape(1, -1)  # Reshape to 2D array with a single sample

# Initialize and train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_flat, y)

# Predict the single target value
prediction = model.predict(X_flat)

# Print the prediction result
print("\nOriginal Output :")
print(y[0])
print("\nPredicted Output for Single Target:")
print(prediction[0])
