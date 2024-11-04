import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

input_file = 'input.csv'
output_file = 'output.csv'

df_input = pd.read_csv(input_file)
df_output = pd.read_csv(output_file)

input_columns = df_input.columns
output_columns = df_output.columns

X = df_input[[col for col in input_columns if col != 'SerialNumber' and col != 'Category']]
y = df_output['SumOfProducts']

label_encoder = LabelEncoder()
df_input['Category'] = label_encoder.fit_transform(df_input['Category'])
df_output['Category'] = label_encoder.transform(df_output['Category'])

X = pd.concat([df_input[['Category']], X], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

df_input['Predicted_SumOfProducts'] = model.predict(X)
print(df_input[['Category', 'Predicted_SumOfProducts']])
