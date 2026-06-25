import sys

import pandas as pd
from ovito.io import import_file
import WarrenCowleyParameters as wc
from ovito.modifiers import *

file_path = sys.argv[1]
# file_path = './Save_data/data100_Gen_M1_0_1501.dat'  (example)
# print(file_path)

pipeline = import_file(file_path)

def setup_particle_types(frame, data):
    types = data.particles_.particle_types_
    types.type_by_id_(4).name = ""
    types.type_by_id_(4).radius = 0.0
    types.type_by_id_(4).color = (1, 0.4, 1)

pipeline.modifiers.append(setup_particle_types)

# slice_mod = SliceModifier(normal=(0, 0, 1), distance=17.6706, slab_width=11.1)
# pipeline.modifiers.append(slice_mod)
#
particle_selection = ExpressionSelectionModifier(
    expression='Position.Z>15 && Position.Z<19',
)
pipeline.modifiers.append(particle_selection)
mod = wc.WarrenCowleyParameters(nneigh=[0, 12], only_selected=True)

pipeline.modifiers.append(mod)
data = pipeline.compute()

wc_for_shells = data.attributes["Warren-Cowley parameters"]
print(f"1NN Warren-Cowley parameters: \n {wc_for_shells[0]}")
# print(f"2NN Warren-Cowley parameters: \n {wc_for_shells[1]}")


# Alternatively, can see it as a dictionarry
print(data.attributes["Warren-Cowley parameters by particle name"])

data_dict = data.attributes["Warren-Cowley parameters by particle name"][0]
data_array = data.attributes["Warren-Cowley Concentration"]
data_value = data.attributes["Warren-Cowley Selected_counts"]
data_atom_info = data.attributes["Warren-Cowley Per_atom info"]

df_from_dict = pd.DataFrame(list(data_dict.items()), columns=["Type Combination", "Value"])
df_single_row = pd.DataFrame([data_dict])


def process_debug_info(debug_info):
    # Flatten the list of dictionaries into a DataFrame
    df = pd.DataFrame(debug_info)

    # Expand 'neighbor_counts' into separate columns if needed
    max_neighbors = max(len(entry['neighbor_counts']) for entry in debug_info)
    neighbor_cols = [f"neighbor_count_{i}" for i in range(max_neighbors)]
    neighbor_data = pd.DataFrame(df['neighbor_counts'].to_list(), columns=neighbor_cols)

    # Combine into a single DataFrame
    final_df = pd.concat([df.drop(columns=['neighbor_counts']), neighbor_data], axis=1)
    return final_df

df1 = process_debug_info(data_atom_info)

# Calculate total_neighbors and P_1st values
df1['total_neighbors'] = df1['num_atoms'] * df1['num_neighbors']

# Calculate P_1st for each neighbor_count column
for col in ['neighbor_count_0', 'neighbor_count_1', 'neighbor_count_2', 'neighbor_count_3', 'neighbor_count_4']:
    df1[f'P_{col.split("_")[-1]}'] = df1[col] / df1['total_neighbors']

columns_of_interest = ['P_0', 'P_1', 'P_2', 'P_3', 'P_4']
probabilities_data = df1[columns_of_interest]
# Reshape the data to put all probabilities in one row with combined column names
reshaped_data = probabilities_data.unstack().reset_index()
reshaped_data['combined'] = reshaped_data['level_0'].str.extract(r'(\d)')[0].astype(int) + 1
reshaped_data['combined'] = 'P_' + reshaped_data['combined'].astype(str) + '_' + (reshaped_data['level_1'] + 1).astype(str)
reshaped_row = reshaped_data.set_index('combined')[0].T

# Convert reshaped data to a single-row dataframe
reshaped_row_df = reshaped_row.to_frame().T
# drop the combined column
# reshaped_row_df.drop(columns='combined', inplace=True)

# data_array to DataFrame and transpose then column names Type1, Type2, Type3, Type4, Type5
df_single_1 = pd.DataFrame(data_array).T
df_single_1.columns = ['Type1', 'Type2', 'Type3', 'Type4', 'Type5']


# Combine the dataframes of reshaped_row_df and df_single_row
df_single_row = pd.concat([df_single_row, reshaped_row_df, df_single_1], axis=1)




#df1.to_csv("WC_per_atom.csv", index=False)

df_single_row.to_csv("WC_parameters.csv", index=False)
