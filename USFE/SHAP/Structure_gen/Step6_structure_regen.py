import numpy as np
import torch
import pandas as pd

# load X_top10_indices_mean_sorted_y_zipped.pt
X_Data = torch.load('../X_test_M1_augmented.pt', map_location='cpu')

# read random10_indices_sorted_y_list.txt as a list
with open('../random500_indices_sorted_y_list.txt') as f:
    random500_indices_sorted_y_list = f.read().splitlines()

random500_list = [int(i) for i in random500_indices_sorted_y_list]

# make random500_list a numpy array
random500_array = np.array(random500_list)

X_Data = X_Data[random500_array]

# In[]
# read the data100_use1.dat for the header first 18 lines. keep the empty lines
df_header = pd.read_csv('data_111_use_AM1_9t.dat', header=None, nrows=22, skip_blank_lines=False)

df_header.columns = ['c1']

# In[]
# read the data100_use1.dat for the data skipping the first 18 lines
df_data = pd.read_csv('data_111_use_AM1_9t.dat', header=None, skiprows=22, sep='\s+')
# sort the data by the first column
df_data = df_data.sort_values(by=0)

df_data.columns = ['id', 'type', 'x', 'y', 'z']

# In[]

for i in range(len(X_Data)):
    # use the X_top10_indices_mean_sorted_y for the type
    new_type_temp = X_Data[i]

    # convert the new_type_temp to a list
    new_type_temp_list = new_type_temp.tolist()

    # use the new_type_temp_list to replace the type column
    df_data['type'] = new_type_temp_list

    a = random500_array[i]
    shap_file_name = 'Random_shap_values_bulk_sample_' + str(i) + '_' + str(a) + '.csv'
    # get the path of shap_file_name
    shap_file_name_use = '../Shap_values/' + shap_file_name

    # load shap data
    df_shap = pd.read_csv(shap_file_name_use)
    # sort the shap data by the index column
    df_shap = df_shap.sort_values(by=['index'])
    # index column + 1
    df_shap['index'] = df_shap['index'] + 1

    # merge the df_data and df_shap, df_data use id column, df_shap use index column
    df_data_merged = pd.merge(df_data, df_shap, how='left', left_on='id', right_on='index')
    # delete the index column
    df_data_merged = df_data_merged.drop(['index'], axis=1)

    # Mn Cr Co Fe Ni
    # for type equal to 3, diffenrence < 0, change the type to 6
    df_data_merged.loc[(df_data_merged['type'] == 1) & (df_data_merged['difference'] < -0.5), 'type'] = 6
    df_data_merged.loc[(df_data_merged['type'] == 4) & (df_data_merged['difference'] < -0.5), 'type'] = 7
    # for type equal to 4, diffenrence < 0, change the type to 7
    df_data_merged.loc[(df_data_merged['type'] == 3) & (df_data_merged['difference'] > 0.4), 'type'] = 8
    df_data_merged.loc[(df_data_merged['type'] == 5) & (df_data_merged['difference'] > 0.4), 'type'] = 9

    # save the data
    df_data_merged.to_csv('data100_temp.csv', index=False, header=False, sep=' ')

    # In[]
    # read the data100_temp.csv without the header
    df_data_loaded = pd.read_csv('data100_temp.csv', header=None)
    df_data_loaded.columns = ['c1']

    data_concat = pd.concat([df_header, df_data_loaded], axis=0)

    output_file_name = 'data100_Gen_M1_' + str(i) + '_' + str(a) + '.dat'
    # save the data to Save_data folder
    data_concat.to_csv('./Save_data/' + output_file_name, index=False, header=False)
