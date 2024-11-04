import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np

input_data = pd.read_csv('input.csv')

summary_data = pd.read_csv('summary.csv')


input_data['Product'] = input_data['Value'] * input_data['Random Number']
aggregated_data = input_data.groupby('Category').agg({
    'Value': 'mean',  # Mean of 'Value' for each category
    'Random Number': 'mean',  # Mean of 'Random Number' for each category
    'Product': 'sum'  # Target: Sum of products for each category
}).reset_index().rename(columns={'Product': 'Sum of Products'})

encoder = OneHotEncoder(sparse_output=False, drop='first')
category_encoded = encoder.fit_transform(aggregated_data[['Category']])
category_encoded_df = pd.DataFrame(category_encoded, columns=encoder.get_feature_names_out(['Category']))

features = pd.concat([aggregated_data[['Value', 'Random Number']], category_encoded_df], axis=1)
target = aggregated_data['Sum of Products']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)

comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison_df)
