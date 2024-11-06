import pandas as pd
import numpy as np

num_rows = 100000
categories = ['A', 'B']  
data = {
    'Category': np.random.choice(categories, size=num_rows),
    'Random_Number_1': np.random.uniform(0.1, 0.9, size=num_rows),
    'Random_Number_2': np.random.randint(1000000, 1200000, size=num_rows),
}

data['Product'] = data['Random_Number_1'] * data['Random_Number_2']

df = pd.DataFrame(data)
print(df.head())
print("------------------------------------------------------------------------------")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error

X = df[['Random_Number_1', 'Random_Number_2']]
y = df['Product']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
mae = mean_absolute_error(y_test, y_pred_tree)
print("Mean Absolute Error:", mae)
print("------------------------------------------------------------------------------")
new_data = {
    'Category': ['A', 'B', 'A', 'A'],
    'Random_Number_1': [0.4, 0.6, 0.7, 0.3],
    'Random_Number_2': [1100000, 1050000, 1150000, 1020000]
}
new_df = pd.DataFrame(new_data)
new_df['Predicted_Product'] = model.predict(new_df[['Random_Number_1', 'Random_Number_2']])
print(new_df)
new_df['Product'] = new_df['Random_Number_1'] * new_df['Random_Number_2']
print(new_df['Product'])

print("------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------")
print("------------------------------------------------------------------------------")



from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
tree_model = DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
mse_tree = mean_squared_error(y_test, y_pred_tree)
print("Decision Tree Mean Squared Error:", mse_tree)
mae = mean_absolute_error(y_test, y_pred_tree)
print("Decision Tree Mean Absolute Error:", mae)
print("------------------------------------------------------------------------------")
new_data = {
    'Category': ['A', 'C', 'A', 'D'],
    'Random_Number_1': [0.4, 0.6, 0.7, 0.3],
    'Random_Number_2': [1100000, 1050000, 1150000, 1020000]
}
new_df = pd.DataFrame(new_data)
new_df['Predicted_Product'] = tree_model.predict(new_df[['Random_Number_1', 'Random_Number_2']])
print(new_df)
new_df['Product'] = new_df['Random_Number_1'] * new_df['Random_Number_2']
print(new_df['Product'])


from sklearn.ensemble import RandomForestRegressor
forest_model = RandomForestRegressor(random_state=42)
forest_model.fit(X_train, y_train)
y_pred_forest = forest_model.predict(X_test)
mse_forest = mean_squared_error(y_test, y_pred_forest)
print("Random Forest Mean Squared Error:", mse_forest)
mae = mean_absolute_error(y_test, y_pred_tree)
print("Random Forest Mean Absolute Error:", mae)
print("------------------------------------------------------------------------------")
# Example new data with 3 columns (Category, Random_Number_1, Random_Number_2)
new_data = {
    'Category': ['A', 'C', 'A', 'D'],
    'Random_Number_1': [0.4, 0.6, 0.7, 0.3],
    'Random_Number_2': [1100000, 1050000, 1150000, 1020000]
}
new_df = pd.DataFrame(new_data)
new_df['Predicted_Product'] = forest_model.predict(new_df[['Random_Number_1', 'Random_Number_2']])
print(new_df)
new_df['Product'] = new_df['Random_Number_1'] * new_df['Random_Number_2']
print(new_df['Product'])


from sklearn.ensemble import GradientBoostingRegressor
# Initialize the model
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
# Predict on the test set
y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
print("Gradient Boosting Mean Squared Error:", mse_gb)
mae = mean_absolute_error(y_test, y_pred_tree)
print("Gradient Boosting Mean Absolute Error:", mae)
print("------------------------------------------------------------------------------")
# Example new data with 3 columns (Category, Random_Number_1, Random_Number_2)
new_data = {
    'Category': ['A', 'C', 'A', 'D'],
    'Random_Number_1': [0.4, 0.6, 0.7, 0.3],
    'Random_Number_2': [1100000, 1050000, 1150000, 1020000]
}
new_df = pd.DataFrame(new_data)
new_df['Predicted_Product'] = gb_model.predict(new_df[['Random_Number_1', 'Random_Number_2']])
print(new_df)
new_df['Product'] = new_df['Random_Number_1'] * new_df['Random_Number_2']
print(new_df['Product'])


from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
svr_model = SVR()
svr_model.fit(X_train_scaled, y_train)
y_pred_svr = svr_model.predict(X_test_scaled)
mse_svr = mean_squared_error(y_test, y_pred_svr)
print("Support Vector Regressor Mean Squared Error:", mse_svr)
mae = mean_absolute_error(y_test, y_pred_tree)
print("Support Vector Regressor Mean Absolute Error:", mae)
print("------------------------------------------------------------------------------")
# Example new data with 3 columns (Category, Random_Number_1, Random_Number_2)
new_data = {
    'Category': ['A', 'C', 'A', 'D'],
    'Random_Number_1': [0.4, 0.6, 0.7, 0.3],
    'Random_Number_2': [1100000, 1050000, 1150000, 1020000]
}
new_df = pd.DataFrame(new_data)
new_df['Predicted_Product'] = svr_model.predict(new_df[['Random_Number_1', 'Random_Number_2']])
print(new_df)
new_df['Product'] = new_df['Random_Number_1'] * new_df['Random_Number_2']
print(new_df['Product'])


from sklearn.neural_network import MLPRegressor
# Initialize the model
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp_model.fit(X_train, y_train)
# Predict on the test set
y_pred_mlp = mlp_model.predict(X_test)
mse_mlp = mean_squared_error(y_test, y_pred_mlp)
print("Neural Network Mean Squared Error:", mse_mlp)
mae = mean_absolute_error(y_test, y_pred_tree)
print("Neural Network Mean Absolute Error:", mae)
print("------------------------------------------------------------------------------")
# Example new data with 3 columns (Category, Random_Number_1, Random_Number_2)
new_data = {
    'Category': ['A', 'C', 'A', 'D'],
    'Random_Number_1': [0.4, 0.6, 0.7, 0.3],
    'Random_Number_2': [1100000, 1050000, 1150000, 1020000]
}
new_df = pd.DataFrame(new_data)
new_df['Predicted_Product'] = mlp_model.predict(new_df[['Random_Number_1', 'Random_Number_2']])
print(new_df)
new_df['Product'] = new_df['Random_Number_1'] * new_df['Random_Number_2']
print(new_df['Product'])

