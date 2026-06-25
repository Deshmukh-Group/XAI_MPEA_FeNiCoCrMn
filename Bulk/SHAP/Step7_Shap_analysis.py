# find all the files in Save_data folder
import os

from tqdm import tqdm

path = "./Structure_gen/Save_data"

files = os.listdir(path)
files.sort()

# for file in files:
#     # print file name
#     print(file)

# load file using pandas
import pandas as pd

# data1 is empty dataframe with columns id, type, x, y, z, positive, negative, diff
data1 = pd.DataFrame(columns=['id', 'type', 'x', 'y', 'z', 'positive', 'negative', 'diff'])
data2 = pd.DataFrame(columns=['id', 'type', 'x', 'y', 'z', 'positive', 'negative', 'diff'])
data3 = pd.DataFrame(columns=['id', 'type', 'x', 'y', 'z', 'positive', 'negative', 'diff'])
data4 = pd.DataFrame(columns=['id', 'type', 'x', 'y', 'z', 'positive', 'negative', 'diff'])
data5 = pd.DataFrame(columns=['id', 'type', 'x', 'y', 'z', 'positive', 'negative', 'diff'])

for file in tqdm(files):
    data = pd.read_csv(path + "/" + file, sep=r'\s+', header=None, skiprows=18)
    data.columns = ['id', 'type', 'x', 'y', 'z', 'positive', 'negative', 'diff']

    # sort data by type and diff in ascending order
    data_new = data.sort_values(by=['type', 'diff'], ascending=[True, True])

    # get the rows type 1
    data_new_1 = data_new[data_new['type'] == 1]
    # get the rows type 2
    data_new_2 = data_new[data_new['type'] == 2]
    # get the rows type 3
    data_new_3 = data_new[data_new['type'] == 3]
    # get the rows type 4
    data_new_4 = data_new[data_new['type'] == 4]
    # get the rows type 5
    data_new_5 = data_new[data_new['type'] == 5]

    # concatenate the data_new_1 and data1
    data1 = pd.concat([data1, data_new_1], axis=0)
    # concatenate the data_new_2 and data2
    data2 = pd.concat([data2, data_new_2], axis=0)
    # concatenate the data_new_3 and data3
    data3 = pd.concat([data3, data_new_3], axis=0)
    # concatenate the data_new_4 and data4
    data4 = pd.concat([data4, data_new_4], axis=0)
    # concatenate the data_new_5 and data5
    data5 = pd.concat([data5, data_new_5], axis=0)


# In[]

# plot the data1 histogram for diff
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set_style("white")

# Define your data and labels
datasets = {
    'Mn': data1['diff'],
    'Cr': data2['diff'],
    'Co': data3['diff'],  # Example label for data3
    'Fe': data4['diff'],  # Example label for data4
    'Ni': data5['diff']   # Example label for data5
}

# In[]
# Create the histograms
for label, data in datasets.items():
    sns.histplot(data, bins=1000, fill=True, common_norm=False, kde=True, label=label, legend=True)

# Add a dashed vertical line at x = 0
plt.axvline(x=0, color='black', linestyle='--')
# plt.axvline(x=-0.5, color='black', linestyle='--')
plt.axvline(x=0.01, color='black', linestyle='--')

# Label the x-axis and y-axis
plt.xlabel('SHAP values')
plt.ylabel('Frequency')

# ylim 0 to 3000
# plt.ylim(0, 3000)
plt.xlim(-0.06, 0.03)

# Add a legend
plt.legend(title='Elements')

# Optionally, add a title
# plt.title('Distribution of SHAP values')

# save the figure as png
plt.savefig('data31.png', dpi=300)
# plt.show()
