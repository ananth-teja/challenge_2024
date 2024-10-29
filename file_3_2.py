import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

data = {
    'CAP_ID': range(1000, 2000),
    'GLA_NO': np.random.choice(['A', 'B'], size=1000),
    'PD': np.random.uniform(0.1, 0.9, size=1000),
    'EAD': np.random.randint(500, 1000, size=1000)
}
 
df = pd.DataFrame(data)

df['GLA_NO'] = df['GLA_NO'].apply(lambda x: 0 if x == 'A' else 1)

X = df[['PD', 'EAD', 'GLA_NO']]
y_A = (df['GLA_NO'] == 0) * df['PD'] * df['EAD']
y_B = (df['GLA_NO'] == 1) * df['PD'] * df['EAD']

X_train, X_test, y_A_train, y_A_test = train_test_split(X, y_A, test_size=0.2, random_state=42)
_, _, y_B_train, y_B_test = train_test_split(X, y_B, test_size=0.2, random_state=42)

sum_A = (y_A_test).sum()  
sum_B = (y_B_test).sum()  

output = pd.DataFrame({
    'GLA_NO': ['A', 'B'],
    'SUM_B3A': [sum_A, sum_B]
})

print(output)

print("---------------------------------------------------------")
dt_model_A = DecisionTreeRegressor()
dt_model_A.fit(X_train, y_A_train)
y_A_pred_dt = dt_model_A.predict(X_test)
mse_A_dt = mean_squared_error(y_A_test, y_A_pred_dt)

dt_model_B = DecisionTreeRegressor()
dt_model_B.fit(X_train, y_B_train)
y_B_pred_dt = dt_model_B.predict(X_test)
mse_B_dt = mean_squared_error(y_B_test, y_B_pred_dt)

sum_A = y_A_pred_dt.sum()  
sum_B = y_B_pred_dt.sum()

output = pd.DataFrame({
    'GLA_NO': ['A', 'B'],
    'SUM_B3A': [sum_A, sum_B]
})

print(output)

print("---------------------------------------------------------")

rf_model_A = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_A.fit(X_train, y_A_train)
y_A_pred_rf = rf_model_A.predict(X_test)
mse_A_rf = mean_squared_error(y_A_test, y_A_pred_rf)

rf_model_B = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model_B.fit(X_train, y_B_train)
y_B_pred_rf = rf_model_B.predict(X_test)
mse_B_rf = mean_squared_error(y_B_test, y_B_pred_rf)

sum_A = y_A_pred_rf.sum()  
sum_B = y_B_pred_rf.sum()

output = pd.DataFrame({
    'GLA_NO': ['A', 'B'],
    'SUM_B3A': [sum_A, sum_B]
})

print(output)

print("---------------------------------------------------------")

gb_model_A = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model_A.fit(X_train, y_A_train)
y_A_pred_gb = gb_model_A.predict(X_test)
mse_A_gb = mean_squared_error(y_A_test, y_A_pred_gb)

gb_model_B = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model_B.fit(X_train, y_B_train)
y_B_pred_gb = gb_model_B.predict(X_test)
mse_B_gb = mean_squared_error(y_B_test, y_B_pred_gb)


sum_A = y_A_pred_gb.sum()  
sum_B = y_B_pred_gb.sum()

output = pd.DataFrame({
    'GLA_NO': ['A', 'B'],
    'SUM_B3A': [sum_A, sum_B]
})

print(output)

print("---------------------------------------------------------")

print(f'Decision Tree Model A MSE: {mse_A_dt}, Model B MSE: {mse_B_dt}')
print(f'Random Forest Model A MSE: {mse_A_rf}, Model B MSE: {mse_B_rf}')
print(f'Gradient Boosting Model A MSE: {mse_A_gb}, Model B MSE: {mse_B_gb}')