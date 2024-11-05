import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
import glob
import os

input_dir = 'path_to_input_files/'
output_dir = 'path_to_output_files/'
models = []

# Hyperparameters
input_shape = (None, number_of_input_features)  # None allows for variable sequence length
batch_size = 32
epochs = 10

# Loop over each input-output pair
for i, input_file in enumerate(glob.glob(os.path.join(input_dir, '*.csv'))):
    input_data = pd.read_csv(input_file).values
    output_file = os.path.join(output_dir, f'output_{i+1}.csv')
    output_data = pd.read_csv(output_file).values.flatten()  # Assuming single output value per file

    # Create a model for this input-output pair
    model = Sequential([
        Masking(mask_value=0.0, input_shape=input_shape),  # Optional masking for padding
        LSTM(64, activation='relu', return_sequences=False),  # LSTM layer
        Dense(32, activation='relu'),
        Dense(1)  # Output layer with one unit for single value prediction
    ])

    model.compile(optimizer=Adam(), loss='mse')

    # Reshape output to match batch dimension
    y_train = np.array(output_data).reshape(-1, 1)

    # Train the model
    model.fit(np.expand_dims(input_data, axis=0), y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    models.append(model)
    model.save(f'model_{i+1}.h5')

print("Models trained for each input-output pair.")

# Prediction on new data
new_input_file = 'path_to_new_input_file.csv'
new_data = pd.read_csv(new_input_file).values

predictions = []
for i in range(1, 11):
    model = keras.models.load_model(f'model_{i}.h5')
    pred = model.predict(np.expand_dims(new_data, axis=0))
    predictions.append(pred)

final_prediction = np.mean(predictions)

print("Final aggregated prediction:", final_prediction)
