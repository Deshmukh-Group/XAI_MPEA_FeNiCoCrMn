import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Load the second file without a header
mpea_file_path = 'MPEA_data_aug_1.csv'
mpea_data = pd.read_csv(mpea_file_path, header=None)
mpea_data.columns = ['Ori_index', 'M_index']
# Extract the second column from the MPEA data for sorting
sort_order = mpea_data['M_index']


def process_data(file_path):
    data = pd.read_csv(file_path)

    data['index_new'] = data['index'] + 1

    # Sort the first file based on the index column according to the sort_order
    sorted_data = data.set_index('index_new').reindex(sort_order).reset_index()

    grouped_stats_corrected = (
        sorted_data['difference']
        .groupby(sorted_data.index // 200)
        .agg(['sum', lambda x: x.abs().mean()])
        .reset_index(drop=True)
    )

    # Rename columns for clarity
    grouped_stats_corrected.columns = ['Sum of Differences', 'Mean Absolute Difference']

    # layer 9 (row 1601 to 1800)
    # layer 10 (row 1801 to 2000)

    layer_10_data = sorted_data.iloc[1800:2000]
    # sum of differences for layer 10
    layer_10_sum = layer_10_data['difference'].sum()
    # mean absolute difference for layer 10
    layer_10_mean = layer_10_data['difference'].abs().mean()

    # Extract data for plotting
    layers = grouped_stats_corrected.index + 1  # Layer numbers (1-based)
    sums = grouped_stats_corrected['Sum of Differences']
    means = grouped_stats_corrected['Mean Absolute Difference']
    abs_sums = sums.abs()

    return grouped_stats_corrected, layer_10_data, layer_10_sum, layer_10_mean, layers, sums, means, abs_sums


# grouped_stats_corrected, layer_10_data, layer_10_sum, layer_10_mean, layers, sums, means, abs_sums = process_data('../Shap_values/Random_shap_values_bulk_sample_2_3092.csv')

# Define the directory and read all files
directory = '../Shap_values/'
# sort_order = list(range(1, 3100))  # Assuming the sort order is a range from 1 to 3100

# Initialize an empty dictionary to store DataFrames
processed_files = {}

for file_name in os.listdir(directory):
    if file_name.endswith('.csv'):
        file_path = os.path.join(directory, file_name)
        try:
            results = process_data(file_path)
            processed_files[file_name] = results
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

# grouped_stats_corrected, layer_10_data, layer_10_sum, layer_10_mean, layers, sums, means, abs_sums = processed_files[
#     'Random_shap_values_bulk_sample_0_860.csv']

# Iterate over each file in the processed_files dictionary and extract the values
grouped_stats_summary = []
abs_sums_dict = {}
means_500 = {}

for file_name, file_data in processed_files.items():
    try:
        (
            grouped_stats_corrected,
            layer_10_data,
            layer_10_sum,
            layer_10_mean,
            layers,
            sums,
            means,
            abs_sums
        ) = file_data

        # Append a summary of the grouped stats for each file
        grouped_stats_summary.append({
            "File Name": file_name,
            "Layer 10 Sum": layer_10_sum,
            "Layer 10 Mean": layer_10_mean,
        })
        # Convert the Series abs_sums to a DataFrame and add the filename as the column name
        abs_sums_dict[file_name] = abs_sums.rename(file_name)
        means_500[file_name] = means.rename(file_name)
    except Exception as e:
        print(f"Error processing file {file_name}: {e}")

# Convert the summary into a DataFrame and display it to the user
grouped_stats_df = pd.DataFrame(grouped_stats_summary)

# Concatenate all dataframes along the columns to create a (18, 500) dataframe
if abs_sums_dict:
    concatenated_abs_sums_df = pd.concat(abs_sums_dict.values(), axis=1)
else:
    concatenated_abs_sums_df = pd.DataFrame()

if means_500:
    concatenated_means_df = pd.concat(means_500.values(), axis=1)
else:
    concatenated_means_df = pd.DataFrame()

# Check if the concatenated_abs_sums_df exists and is not empty
if 'concatenated_abs_sums_df' in locals() and not concatenated_abs_sums_df.empty:
    # Calculate the mean and standard deviation for each row
    concatenated_abs_sums_df['Mean'] = concatenated_abs_sums_df.mean(axis=1)
    concatenated_abs_sums_df['Std'] = concatenated_abs_sums_df.std(axis=1)

else:
    print("The concatenated_abs_sums_df is not available or is empty.")

# Check if the concatenated_means_df exists and is not empty
if 'concatenated_means_df' in locals() and not concatenated_means_df.empty:
    # Calculate the mean and standard deviation for each row
    concatenated_means_df['Mean'] = concatenated_means_df.mean(axis=1)
    concatenated_means_df['Std'] = concatenated_means_df.std(axis=1)

# In[]


means = concatenated_abs_sums_df['Mean']
stds = concatenated_abs_sums_df['Std']
layers = concatenated_abs_sums_df.index + 1

means_L = concatenated_means_df['Mean']
stds_L = concatenated_means_df['Std']

# Define font size
font_size = 18

# Update font properties globally
plt.rcParams.update({
    'font.size': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'axes.titlesize': font_size,
    'axes.labelsize': font_size,
    'legend.fontsize': font_size
})

# Create a figure with two subplots and enable constrained_layout for better alignment
fig = plt.figure(figsize=(14, 8), constrained_layout=True)
gs = GridSpec(2, 1, height_ratios=[1, 2], figure=fig)

# Plot Mean Absolute Difference (MAD) on the top subplot
ax1 = fig.add_subplot(gs[1])
ax1.bar(layers, means, yerr=stds, capsize=10, color='red', label='Layer-wise\nMean Absolute SHAP Value',
        linestyle='--')
ax1.set_ylabel('Mean Absolute\nSHAP Value')
ax1.grid(axis='y', linestyle='--', alpha=0.6)

max_mean = max(concatenated_abs_sums_df['Mean']) + max(concatenated_abs_sums_df['Std'])
import math

rounded_max_mean = math.ceil(max_mean * 10) / 10
ax1.set_ylim(0, 25)
ax1.set_yticks(np.arange(0, rounded_max_mean + 0.01, 4))  # Set Y-ticks for MAD on ax1
ax1.legend(loc='upper right')
ax1.tick_params(direction='in')  # Hide X-ticks for ax1

# Plot Sum of Differences as a bar chart on the bottom subplot
ax2 = fig.add_subplot(gs[0], sharex=ax1)
ax2.errorbar(layers, means_L, yerr=stds_L, capsize=8, fmt='D', color='orange', alpha=0.7,
         label='Atom-wise\nMean Absolute SHAP Value')
ax1.set_xlabel('Layer Number')
ax2.set_ylabel('Mean Absolute\nSHAP Value')
ax2.set_xticks(range(1, len(layers) + 1))  # Set X-ticks for layers on ax2
ax2.tick_params(labelbottom=False, direction='in')  # Ensure X-ticks are visible
ax2.grid(axis='y', linestyle='--', alpha=0.6)
ax2.set_yticks(np.arange(0, 0.51, 0.1))
ax2.set_ylim(0, 0.6)

# Custom legend for red and blue bars
ax2.legend(loc='upper right')

# Adjust layout and ensure proper alignment
# plt.tight_layout()
# plt.show()
# Save the figure as a PNG file
plt.savefig('layer_analysis_MS1_Overall.png', dpi=300)
