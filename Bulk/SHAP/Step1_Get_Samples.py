import torch
import numpy as np

X_data = 'X_test_M1_augmented.pt'
y_data = 'y_test.pt'

X = torch.load(str(X_data), map_location=torch.device('cpu'))
y = torch.load(str(y_data), map_location=torch.device('cpu'))

# In[]
# get random 10 of the first column of y and its indices using random.choice with defined seed
np.random.seed(42)
random10_indices_sorted_y = np.random.choice(np.arange(0, y.shape[0]), 500, replace=False)

# check if the random10_indices_sorted_y has duplicates
if len(random10_indices_sorted_y) != len(set(random10_indices_sorted_y)):
    print('Duplicates exist in random10_indices_sorted_y')

random10_values_sorted_y = y[random10_indices_sorted_y, 0]

# get X data with the same indices as top10_indices_mean_sorted_y
X_random10_indices_sorted_y = X[random10_indices_sorted_y]

# zip the X data and the y data
X_random10_indices_sorted_y_zipped = zip(X_random10_indices_sorted_y, random10_values_sorted_y)

# save the zipped data
# torch.save(X_random10_indices_sorted_y_zipped, 'X_random10_indices_sorted_y_zipped.pt')

# save random10_indices_sorted_y as a list to a file
random10_indices_sorted_y_list = random10_indices_sorted_y.tolist()
with open('random500_indices_sorted_y_list.txt', 'w') as f:
    for item in random10_indices_sorted_y_list:
        f.write("%s\n" % item)
