import torch
import numpy as np

# X_data = 'X_test_M1_augmented.pt'
y_data = 'y_test.pt'

# X = torch.load(str(X_data), map_location=torch.device('cpu'))
y = torch.load(str(y_data), map_location=torch.device('cpu'))

# Step 2: Read indices from the text file
with open('random500_indices_sorted_y_list.txt', 'r') as f:
    random500_indices_sorted_y_list = [int(line.strip()) for line in f.readlines()]

# Step 3: Get the corresponding y values using the indices
random500_y_values = y[random500_indices_sorted_y_list, 0]

# convert y data to another unit
A = 44.054073e-10 * 25.43463e-10 # Mn
eV_to_J = 1.6021766208e-19

y = y / A * eV_to_J * 1000
random500_y_values = random500_y_values / A * eV_to_J * 1000

import pandas as pd

# Step 4: Create a DataFrame with indices and their corresponding y values
data = {
    'Index': random500_indices_sorted_y_list,
    'Y_Value': [y_val.item() for y_val in random500_y_values]
}

df = pd.DataFrame(data)

# Save the DataFrame to a file (optional, e.g., CSV)
df.to_csv('indices_y_values.csv', index=False)

# Display the DataFrame
print(df)
