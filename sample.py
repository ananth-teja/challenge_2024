import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import glob
import os

# Directories containing input and output files
input_dir = 'path_to_input_files/'
output_dir = 'path_to_output_files/'

# Initialize a dictionary to store trained models
models = {}

# Loop over each input-output file pair
for i, input_file in enumerate(glob.glob(os.path.join(input_dir, '*.csv'))):
    # Read the input file
    input_data = pd.read_csv(input_file)
    
    # Corresponding output file
    output_file = os.path.join(output_dir, f'output_{i+1}.csv')  # Adjust as per naming convention
    output_data = pd.read_csv(output_file)

    # Assuming the target column is the last one in the output file
    y = output_data.iloc[:, -1]  # Target
    X = input_data  # Features

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Store the model
    models[f'model_{i+1}'] = model
    joblib.dump(model, f'model_{i+1}.joblib')  # Save each model

    print(f"Trained model {i+1} with MSE: {mean_squared_error(y, model.predict(X))}")

# Load new input file
new_input_file = 'path_to_new_input_file.csv'
new_data = pd.read_csv(new_input_file)

# Aggregate predictions from each model
predictions = []

for i in range(1, 11):  # Assuming there are 10 models
    # Load the model
    model = joblib.load(f'model_{i}.joblib')
    
    # Make prediction on the new input
    pred = model.predict(new_data)
    predictions.append(pred)

# Combine predictions, e.g., by averaging
final_prediction = sum(predictions) / len(predictions)

print("Final aggregated prediction:", final_prediction)

