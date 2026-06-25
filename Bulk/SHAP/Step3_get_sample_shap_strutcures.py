import os

import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm

# load the model from disk
shap_values = joblib.load('shap_values.pkl')

# In[]

shap_values_use = shap_values.values

# for bulk modulus
shap_values_bulk = shap_values_use[0]


# Function to calculate per-structure SHAP value
def get_per_structure_shap(shap_values_bulk_sample):
    """
    Calculate the total SHAP value for a structure by summing absolute SHAP values across all atoms.
    """
    # Sum the absolute SHAP values across all atoms
    per_structure_shap_value = np.sum(shap_values_bulk_sample)
    return per_structure_shap_value


# Read indices of the 500 selected structures
with open('random500_indices_sorted_y_list.txt') as f:
    random500_indices_sorted_y_list = f.read().splitlines()

random500_list = [int(i) for i in random500_indices_sorted_y_list]

# Create a DataFrame to store the results
results = []

# Loop through the selected structures
for i in range(len(random500_list)):
    k = random500_list[i]
    shap_values_bulk_sample = shap_values_bulk[k]  # SHAP values for the selected structure
    per_structure_shap_value = get_per_structure_shap(shap_values_bulk_sample)

    # Append results to the list
    results.append({
        "Structure_Index": k,
        "Per_Structure_SHAP": per_structure_shap_value
    })

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv('Per_Structure_SHAP_Values.csv', index=False)

# Display the DataFrame
print(results_df)
