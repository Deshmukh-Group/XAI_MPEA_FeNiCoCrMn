import torch
import pandas as pd

# Reloading the tensor with the map_location set to CPU
X_train = torch.load('../X_train.pt', map_location='cpu')
X_val = torch.load('../X_val.pt', map_location='cpu')
X_test = torch.load('../X_test.pt', map_location='cpu')

# In[]

df_no_header = pd.read_csv('MPEA_data_aug_1.csv', header=None, names=['Column_1', 'Column_2'])
df_sorted = df_no_header.sort_values(by='Column_1', ascending=True)
# Convert Column_2 into zero-based indices
sorted_indices = df_sorted['Column_2'].argsort()

# # Rearrange each data point in the tensor according to the new positions
X_train_reorder = X_train[:, sorted_indices]
X_val_reorder = X_val[:, sorted_indices]
X_test_reorder = X_test[:, sorted_indices]

# print the first 1 data point in the tensor
print('train:', X_train[0])
print('train_reorder:', X_train_reorder[0])

# # Save the reordered tensors
torch.save(X_train_reorder, './X_train_M1_augmented.pt')
torch.save(X_val_reorder, './X_val_M1_augmented.pt')
torch.save(X_test_reorder, './X_test_M1_augmented.pt')
print('Data augmentation completed successfully!')
