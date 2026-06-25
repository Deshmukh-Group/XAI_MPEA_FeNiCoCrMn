import pandas as pd
from matplotlib import pyplot as plt

# Load the CSV files
wc_parameters_file = 'WC_new_full_parameters_structures.csv'
merged_indices_file = 'merged_indices_y_values_per_structure_shap.csv'

wc_parameters_df = pd.read_csv(wc_parameters_file)
merged_indices_df = pd.read_csv(merged_indices_file)

# Merge the dataframes on the specified columns
merged_df = pd.merge(merged_indices_df, wc_parameters_df,left_on='Index', right_on='W_Index')

# get the Y_predict column top 50
merged_df = merged_df.nlargest(100, 'Y_Actual')

# print the lowest values of Y_Actual in merged_df
print(merged_df.nsmallest(1, 'Y_Actual'))

# get the rows of Type 1 - Type 1 greater small than -0.1
filtered_df = merged_df[merged_df['Type 1-Type 1'] < -0.1]
print(filtered_df)

# show only the 'Type 1-Type 1', 'Type 1-Type 2', 'Type 1-Type 3', 'Type 1-Type 4', 'Type 1-Type 5' of filtered_df
filtered_df = filtered_df[['Type 1-Type 1', 'Type 1-Type 2', 'Type 1-Type 3', 'Type 1-Type 4', 'Type 1-Type 5']]
# values 2 decimal points for filtered_df
filtered_df = filtered_df.round(2)


# check if Index, Structure_Index, and W_Index, are same values
print(merged_df['Index'].equals(merged_df['Structure_Index']))
print(merged_df['Index'].equals(merged_df['W_Index']))


# plt.figure(figsize=(10, 8))
# scatter = plt.scatter(
#     merged_df['Type1'],
#     merged_df['Type 1-Type 1'],
#     c=merged_df['Y_Predict'],
#     cmap='coolwarm',
#     alpha=0.8
# )
# plt.colorbar(scatter, label='Y_Predict')
# plt.xlabel('Concentration')
# plt.ylabel('Warren-Cowley Parameter')
# # plt.grid(True)
# plt.show()
# plt.savefig('Per_Structure_SHAP_vs_Type_4_Type_6.png')

# Extracting data for the updated 5x5 panel plot
type_concentration_columns = ['Type1', 'Type2', 'Type3', 'Type4', 'Type5']
interaction_groups = [
    ['Type 1-Type 1', 'Type 1-Type 2', 'Type 1-Type 3', 'Type 1-Type 4', 'Type 1-Type 5'],
    ['Type 2-Type 1', 'Type 2-Type 2', 'Type 2-Type 3', 'Type 2-Type 4', 'Type 2-Type 5'],
    ['Type 3-Type 1', 'Type 3-Type 2', 'Type 3-Type 3', 'Type 3-Type 4', 'Type 3-Type 5'],
    ['Type 4-Type 1', 'Type 4-Type 2', 'Type 4-Type 3', 'Type 4-Type 4', 'Type 4-Type 5'],
    ['Type 5-Type 1', 'Type 5-Type 2', 'Type 5-Type 3', 'Type 5-Type 4', 'Type 5-Type 5']
]

# for col in interaction_groups find how many values are greate then 0.15 or less then -0.15
for col in interaction_groups:
    for interaction in col:
        count_high = (merged_df[interaction] > 0.1).sum()
        count_low = (merged_df[interaction] < -0.1).sum()
        print(f'{interaction}: >0.1 count = {count_high}, <-0.1 count = {count_low}')

# Setting up the 5x5 grid plot
fig, axes = plt.subplots(5, 5, figsize=(10, 10), sharex=True, sharey=True)

# Creating each column with corresponding interactions
for col_idx, ax_col in enumerate(axes.T):
    concentration_column = type_concentration_columns[col_idx]
    interaction_columns = interaction_groups[col_idx]
    for row_idx, ax in enumerate(ax_col):
        interaction_column = interaction_columns[row_idx]
        scatter = ax.scatter(
            merged_df[concentration_column],
            merged_df[interaction_column],
            c=merged_df['Y_Predict'],
            cmap='Reds',
            alpha=0.8,
            s=10,
            edgecolors='black',
            linewidths=0.2
        )
        ax.set_xlim(0, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_box_aspect(1)
        ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(direction='in')
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        # if row_idx == 4:
            # ax.set_xlabel(f'Concentration ({concentration_column})', fontsize=8)
        # if col_idx == 0:
            # ax.set_ylabel('Warren-Cowley Parameter', fontsize=8)
        # ax.set_title(interaction_column, fontsize=8)

# Adding a colorbar
# cbar = fig.colorbar(scatter, ax=axes, location='right', shrink=0.6, pad=0.1)
# cbar.set_label('Y_Predict', fontsize=10)

# Adjust layout and add colorbar
fig.subplots_adjust(right=0.85)  # Adjust space for the colorbar
cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # Position for the colorbar
cbar = fig.colorbar(scatter, cax=cbar_ax)
cbar.set_label('Bulk Modulus Predicted', fontsize=10)

# Adjust layout
# plt.tight_layout()

plt.savefig('5x5_panel_plot_top100.png', dpi=300)
plt.show()
