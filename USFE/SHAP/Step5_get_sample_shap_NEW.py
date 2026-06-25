import os

import numpy as np
import joblib
from tqdm import tqdm

# load the model from disk
shap_values = joblib.load('shap_values.pkl')

# In[]

shap_values_use = shap_values.values

# for bulk modulus
shap_values_bulk = shap_values_use


def get_shap(shap_values_bulk_sample):
    # shap_values_bulk_max_abs = np.sum(np.abs(shap_values_bulk_sample), axis=0)
    #
    # # Summing positive and negative values separately for each column
    # positive_sum = np.sum(np.where(shap_values_bulk_sample >= 0, shap_values_bulk_sample, 0), axis=0)
    # negative_sum = np.sum(np.where(shap_values_bulk_sample < 0, shap_values_bulk_sample, 0), axis=0)

    common = np.arange(0, 3600)

    shap_values_bulk_sample_index_list = shap_values_bulk_sample[:, common]

    import pandas as pd
    shap_values_bulk_sample_index_list_1 = pd.DataFrame(shap_values_bulk_sample_index_list)
    shap_values_bulk_sample_index_list_1.columns = common.tolist()

    # get the sum of positive and negative values for each column in shap_values_bulk_sample_index_list_1 using pandas
    positive_sum = shap_values_bulk_sample_index_list_1[shap_values_bulk_sample_index_list_1 > 0].sum(axis=0)
    negative_sum = shap_values_bulk_sample_index_list_1[shap_values_bulk_sample_index_list_1 < 0].sum(axis=0)
    # concatenate the positive_sum and negative_sum into a dataframe
    positive_negative_sum = pd.concat([positive_sum, negative_sum], axis=1)
    # use the index of positive_negative_sum as a new column
    positive_negative_sum['index'] = positive_negative_sum.index
    # rename the columns
    positive_negative_sum.columns = ['positive_sum', 'negative_sum', 'index']
    # reset the index
    positive_negative_sum = positive_negative_sum.reset_index(drop=True)
    # new column difference between positive_sum and absolute value of negative_sum
    positive_negative_sum['difference'] = positive_negative_sum['positive_sum'] - abs(
        positive_negative_sum['negative_sum'])
    # sort the dataframe by the difference column
    positive_negative_sum = positive_negative_sum.sort_values(by=['difference'], ascending=False)

    return positive_negative_sum


# In[]

# read random10_indices_sorted_y_list.txt as a list
with open('random500_indices_sorted_y_list.txt') as f:
    random500_indices_sorted_y_list = f.read().splitlines()

random500_list = [int(i) for i in random500_indices_sorted_y_list]

#
for i in range(len(random500_list)):
    k = random500_list[i]
    shap_values_bulk_sample = shap_values_bulk[k]
    positive_negative_sum = get_shap(shap_values_bulk_sample)
    # save to csv
    positive_negative_sum.to_csv('./Shap_values/' + 'Random_shap_values_bulk_sample_' + str(i) + '_' + str(k) + '.csv',
                                 index=False)
    # if the file exists, print the file name
    if os.path.isfile('./Shap_values/' + 'Random_shap_values_bulk_sample_' + str(i) + '_' + str(k) + '.csv'):
        print('OK')
    else:
        print('Random_shap_values_bulk_sample_' + str(i) + '_' + str(k) + '.csv' + ' does not exist')
