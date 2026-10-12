import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    #Input Layer (4 features of flower)
    
    # 2 Hidden Layers
    #Output Layer (3 Classes of Iris Flowers)

    def __init__(self, in_features = 4, h1 = 8, h2 = 8, out_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x 

torch.manual_seed(41)
model = Model()

my_df = pd.read_csv('./Iris.csv')
#Change last column from strings to integers
my_df['Species'] = my_df['Species'].map({
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}).astype(int)

X = my_df.drop(['Id', 'Species'], axis=1)
y = my_df['Species']

X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

#Convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)

#Convert y labels to tensors long
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

#Measures Error
criterion = nn.CrossEntropyLoss()
#Optimizer Adam
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

#Train model
epoch = 100
losses = []
correct = 0

for i in range(epoch):
    y_pred = model.forward(X_train)

    loss = criterion(y_pred, y_train)

    losses.append(loss.detach().numpy())

    if (i % 10 == 0):
        print(f"Epoch: {i} and loss {loss}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(range(epoch), losses)
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.show()

with torch.no_grad():
    for i,data in enumerate(X_test):
        y_val = model.forward(data)

        print(f'{i+1}. {str(y_val)} \t {y_test[i]} \t {y_val.argmax().item()}')

        if (y_val.argmax().item() == y_test[i]):
            correct += 1

print(f"We got {correct} correct!")











