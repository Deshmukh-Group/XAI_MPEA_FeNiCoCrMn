import os
import random

import numpy as np
# In[epoch]:
import torch
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
#from torchmetrics.functional import r2_score, mean_squared_error
import sys
import pandas as pd

# from pytorchtools import EarlyStopping

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_data = 'X_test_M1_augmented.pt'
y_data = 'y_test.pt'

X = torch.load(str(X_data), map_location=torch.device(device))
y = torch.load(str(y_data), map_location=torch.device(device))

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.relu = nn.ReLU()
        self.layer = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, )
        # self.layer1 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, )
        # self.layer2 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, )
        # self.layer3 = nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, )
        self.pooling = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(984, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 5)
        self.embedding = nn.Embedding(6, 8)

    # @torchsnooper.snoop()
    def forward(self, x, **kwargs):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.layer(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.flatten(x)

        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        x = self.linear5(x)
        return x


model = SimpleCNN()
model.to(device)

# Load trained model weights (adjust path as needed)
model.load_state_dict(torch.load('checkpoint.pt', map_location=device))
model.eval()



# Step 2: Read indices from the text file
with open('random500_indices_sorted_y_list.txt', 'r') as f:
    random500_indices_sorted_y_list = [int(line.strip()) for line in f.readlines()]

# Step 3: Get the corresponding y values using the indices
random500_y_values = y[random500_indices_sorted_y_list, 0]

# Step 4: Get the X values corresponding to the indices and predict y_predict
random500_X_values = X[random500_indices_sorted_y_list]
with torch.no_grad():
    random500_y_predict = model(random500_X_values.to(device)).cpu().numpy()

# Step 5: Create a DataFrame with indices, actual y values, and predicted y values
data = {
    'Index': random500_indices_sorted_y_list,
    'Y_Actual': [y_val.item() for y_val in random500_y_values],
    'Y_Predict': [y_pred[0] for y_pred in random500_y_predict]
}

df = pd.DataFrame(data)

# Save the DataFrame to a file (optional, e.g., CSV)
df.to_csv('indices_y_values_predictions.csv', index=False)

# Display the DataFrame
print(df)
