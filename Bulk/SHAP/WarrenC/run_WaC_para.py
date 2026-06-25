import numpy as np
import subprocess

import pandas as pd
from tqdm import tqdm

# read random10_indices_sorted_y_list.txt as a list
with open('../random500_indices_sorted_y_list.txt') as f:
    random500_indices_sorted_y_list = f.read().splitlines()

random500_list = [int(i) for i in random500_indices_sorted_y_list]

# make random500_list a numpy array
random500_array = np.array(random500_list)

for i in tqdm(range(len(random500_list))):
    a = random500_array[i]
    file_name = 'data100_Gen_M1_' + str(i) + '_' + str(a) + '.dat'
    print(file_name, i, a)

    file_path = '../Structure_gen/Save_data/' + file_name

    # excute python script USE_Warren.py, using subprocess
    subprocess.run(['python', 'USE_Warren.py', file_path])

    # change the output file name WC_parameters.csv to WC_parameters_i_a.csv
    subprocess.run(['mv', 'WC_parameters.csv', 'WC_parameters_' + str(i) + '_' + str(a) + '.csv'])

    # read the WC_parameters_i_a.csv as a dataframe
    df_temp = pd.read_csv('WC_parameters_' + str(i) + '_' + str(a) + '.csv')

    # add a column for the index a
    df_temp['W_Index'] = a

    # concatenate the df_temp to df
    if i == 0:
        df = df_temp
    else:
        df = pd.concat([df, df_temp])

    # break

# save the df to a csv file
df.to_csv('WC_new_full_parameters_structures.csv', index=False)
