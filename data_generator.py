import pandas as pd
import numpy as np

# Set the number of records
num_records = 10000

# Generate data for each column
serial_number = np.arange(1, num_records + 1)  # Serial numbers from 1 to num_records
category = np.random.choice(['A', 'B'], num_records)  # Randomly select 'A' or 'B' for each record
values_decimal = np.round(np.random.uniform(0.1, 0.9, num_records), 1)  # Values between 0.1 and 0.9, rounded to 1 decimal
values_range = np.random.randint(100, 501, num_records)  # Integer values between 100 and 500

# Create a DataFrame
df = pd.DataFrame({
    'Serial Number': serial_number,
    'Category': category,
    'Value (0.1 to 0.9)': values_decimal,
    'Value (100 to 500)': values_range
})

# Display the first few rows
print(df.head())


# Calculate the product of the 3rd and 4th columns
df['Product'] = df['Value (0.1 to 0.9)'] * df['Value (100 to 500)']

# Group by category and calculate the sum of the products
category_sums = df.groupby('Category')['Product'].sum()

print("Sum of products by category:")
print(category_sums)
