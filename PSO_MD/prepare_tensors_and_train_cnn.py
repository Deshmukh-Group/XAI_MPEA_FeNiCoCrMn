# coding: utf-8

# In[1]:

import os
import random

import numpy as np
# In[epoch]:
import torch
from matplotlib import pyplot as plt
from pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.functional import r2_score, mean_squared_error

if torch.cuda.is_available():
    device = "cuda:0"
    print("GPU used")
else:
    device = "cpu"
    print("CPU used")

Birds = 200
top = 25000  # (0 for use all data)

# In[2]:

cwd = os.getcwd()

list_temp = []
for i in np.arange(0, Birds, 1):
    list_temp.append(str(i))

list_dir = []
for root, dirs, files in os.walk(cwd):
    for name in dirs:
        if name in list_temp:
            # print(os.path.join(root, name))
            # print(name)
            list_dir.append(os.path.join(root, name))


# In[3]:

def read_type(filename):
    list1 = []
    my_file = open(filename, "r")
    for line in my_file.readlines():
        lines = line.strip()
        # print(line.strip('\n').split(' '))
        list1.append(eval(lines))
    return list1


def read_prop(filename):
    list1 = []
    my_file = open(filename, "r")
    for line in my_file.readlines():
        lines = line.strip('\n').split(' ')
        list1.append(lines)
    return list1


# Python code to sort the tuples using second element
# of sublist Function to sort using sorted()
def Sort(sub_li):
    # reverse = None (Sorts in Ascending order)
    # key is set to sort the first properties
    # sublist lambda has been used
    return sorted(sub_li, key=lambda x: x[1][0], reverse=True)


def list2txt(my_list, filename):
    with open(str(filename) + '.txt', 'w') as f:
        for item in my_list:
            f.write("%s\n" % item)


# In[4]: Prepare Data

X = []
y = []
for bird in list_dir:
    print(bird)
    os.chdir(bird)

    s = read_type('type_file.txt')
    s1 = read_prop('prop.txt')
    if len(s) == len(s1):
        for i in range(0, len(s1)):
            for j in range(0, len(s1[i])):
                s1[i][j] = float(s1[i][j])

        for p in s:
            X.append(p)
        for q in s1:
            y.append(q)
    else:
        print("This bird has mismatch", bird)

os.chdir(cwd)

# In[data clean]
zipped = zip(X, y)
data_unclean = list(zipped)

for element in reversed(data_unclean): # Use reversed because that iterator does not know that a list element was removed
    if element[1][0] == 1000.0 and element[1][1] == 1000.0:  # PSO error output
        data_unclean.remove(element)
        # print(element)

data_clean = Sort(data_unclean)  # 1000.0 removed

if top >= 1:
    data_top = data_clean[0:top]
    X_data = [item[0] for item in data_top]
    y_data = [item[1] for item in data_top]
else:
    X_data = [item[0] for item in data_clean]
    y_data = [item[1] for item in data_clean]


list2txt(y_data, 'Prop_cleaned')


# In[CNN]:


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


# In[GRU]:

# In[6]

model = SimpleCNN()
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

batch_size = 30

# In[24]:

X_data = torch.LongTensor(X_data)
y_data = torch.tensor(y_data, dtype=torch.float)
X_data = X_data.to(device)
y_data = y_data.to(device)

rnd1, rnd2 = random.randint(0, 1000), random.randint(0, 1000)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=rnd1, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=rnd2, test_size=0.1)

dataset_train = torch.utils.data.TensorDataset(X_train, y_train)
train_data = DataLoader(dataset_train, batch_size, shuffle=True)

dataset_val = torch.utils.data.TensorDataset(X_val, y_val)
val_data = DataLoader(dataset_val, batch_size, shuffle=True)

dataset_test = torch.utils.data.TensorDataset(X_test, y_test)
test_data = DataLoader(dataset_test, batch_size, shuffle=True)

# In[Save data]

torch.save(X_train, 'X_train.pt')
torch.save(X_test, 'X_test.pt')
torch.save(X_val, 'X_val.pt')

torch.save(y_train, 'y_train.pt')
torch.save(y_test, 'y_test.pt')
torch.save(y_val, 'y_val.pt')

mean_tmp = torch.mean(y_train, axis=0)
std_tmp = torch.std(y_train, axis=0)
print("Column-wise Train Data Mean:\n", mean_tmp)
print("Column-wise Train Data Standard deviation:\n", std_tmp)
print(torch.max(y_train))

mean_tmp = torch.mean(y_val, axis=0)
std_tmp = torch.std(y_val, axis=0)
print("Column-wise Valid Data Mean:\n", mean_tmp)
print("Column-wise Valid Data Standard deviation:\n", std_tmp)
print(torch.max(y_val))

mean_tmp = torch.mean(y_test, axis=0)
std_tmp = torch.std(y_test, axis=0)
print("Column-wise Test Data Mean:\n", mean_tmp)
print("Column-wise Test Data Standard deviation:\n", std_tmp)
print(torch.max(y_test))

# In[25]:

train_losses = []
valid_losses = []
avg_train_losses = []
avg_valid_losses = []

n_epochs = 50000
patience = 2000
early_stopping = EarlyStopping(patience=patience, verbose=True)

# with torchsnooper.snoop():
for epoch in range(1, n_epochs + 1):
    for x, y in train_data:
        yhat = model(x)
        # loss = torch.mean(torch.sum(torch.pow(yhat - y, 2), dim=1))
        loss = criterion(yhat, y)
        train_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    for x, y in val_data:
        yhat = model(x)
        loss = criterion(yhat, y)
        valid_losses.append(loss.item())

    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)

    epoch_len = len(str(n_epochs))

    print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                 f'train_loss: {train_loss:.5f} ' +
                 f'valid_loss: {valid_loss:.5f}')

    print(print_msg)

    # clear lists to track next epoch
    train_losses = []
    valid_losses = []

    early_stopping(valid_loss, model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# load the last checkpoint with the best model
model.load_state_dict(torch.load('checkpoint.pt'))

# In[] # visualize the loss as the network trained

train_loss = avg_train_losses
valid_loss = avg_valid_losses

list2txt(train_loss, 'train_loss')
list2txt(valid_loss, 'valid_loss')

fig = plt.figure(figsize=(10, 8))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

# find position of lowest validation loss
minposs = valid_loss.index(min(valid_loss)) + 1
plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0, 30)  # consistent scale
plt.xlim(0, len(train_loss) + 1)  # consistent scale
plt.grid(True)
plt.legend()
plt.tight_layout()
# # plt.show()
fig.savefig('loss_plot.png', bbox_inches='tight')

# In[ Testing ]

y_pred = model(X_test)
print(r2_score(y_pred, y_test))
print(mean_squared_error(y_pred, y_test))
