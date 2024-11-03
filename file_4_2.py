import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Step 1: Load Data
input_data = pd.read_csv('input_file.csv')
output_data = pd.read_csv('output_file.csv')

# Merge on 'category' column
merged_data = pd.merge(input_data, output_data, on='category')

# Step 2: Data Preprocessing
X = merged_data[['column1', 'column2', 'column3']]  # Adjust column names
y = merged_data['output_column']  # Replace with the output column name

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection and Training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Step 5: Prediction and Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print("Mean Absolute Error:", mae)
print("Root Mean Squared Error:", rmse)
