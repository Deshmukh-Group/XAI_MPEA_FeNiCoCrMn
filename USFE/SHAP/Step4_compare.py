# Load the uploaded CSV files
import pandas as pd

indices_y_values_path = 'indices_y_values_predictions.csv'
per_structure_shap_values_path = 'Per_Structure_SHAP_Values.csv'

indices_y_values_df = pd.read_csv(indices_y_values_path)
per_structure_shap_values_df = pd.read_csv(per_structure_shap_values_path)

# Merge the two DataFrames using "Structure_Index"
merged_df = pd.merge(indices_y_values_df, per_structure_shap_values_df, left_on="Index", right_on="Structure_Index")
merged_df['Y_Pred_minus_Per_Structure_SHAP'] = merged_df['Y_Predict'] - merged_df['Per_Structure_SHAP']

# merged_df.to_csv('merged_indices_y_values_per_structure_shap.csv', index=False)

# get r2 score using Y_Actual and Y_Predict
from sklearn.metrics import r2_score

r2 = r2_score(merged_df['Y_Actual'], merged_df['Y_Predict'])
print(f"R2 score: {r2}")

# get the range of Y_Pred_minus_Per_Structure_SHAP, mean, std
print(f"Range of Y_Pred_minus_Per_Structure_SHAP: {merged_df['Y_Pred_minus_Per_Structure_SHAP'].min()} - {merged_df['Y_Pred_minus_Per_Structure_SHAP'].max()}")
print(f"Mean of Y_Pred_minus_Per_Structure_SHAP: {merged_df['Y_Pred_minus_Per_Structure_SHAP'].mean()}")
print(f"Standard deviation of Y_Pred_minus_Per_Structure_SHAP: {merged_df['Y_Pred_minus_Per_Structure_SHAP'].std()}")