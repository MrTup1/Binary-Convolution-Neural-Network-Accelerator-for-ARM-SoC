from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

loaders = {
    'train' : DataLoader(train_data,
                         batch_size = 100,
                         shuffle = True,
                         num_workers = 0),

    'test' : DataLoader(test_data,
                         batch_size = 100,
                         shuffle = True,
                         num_workers = 0),
}

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size = 3, stride = 1)
        self.max1 = nn.MaxPool2d(kernel_size = 3, stride = 3)
        self.fc1 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max1(x)
        x = x.view(x.size(0), -1) #Flattens all dimensions except batch size (index 0) for Linear Layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)

criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if (batch_idx % 20 == 0):
            print(f"{batch_idx * len(data)} \t {loss.item():.6f}")

def test():
    model.eval()

    testLoss = 0
    correct = 0

    with torch.no_grad():
        for data,target in loaders['test']:
            data,target = data.to(device), target.to(device)

            output = model(data)
            testLoss += criterion(output, target).item() #.item() used to extract value out of tensor
            
            prediction = output.argmax(dim = 1, keepdim = True)
            correct += prediction.eq(target.view_as(prediction)).sum().item() #target and prediction different formats

    testLoss /= len(loaders['test']) #Average Test Loss for batches
    
    accuracy = 100. * correct / len(loaders['test'].dataset)
    print(f"Average loss: {testLoss:.4f}")
    print(f"Test accuracy: {accuracy:.2f}%")

for epoch in range(10):
    train(epoch)
    test()

## Demonstration
model.eval()
data, target = test_data[1]
data = data.unsqueeze(0).to(device)
output = model(data)
prediction = output.argmax(dim = 1, keepdim = True).item()
print(f"Prediction: {prediction}")
image = data.squeeze(0).squeeze(0).cpu().numpy()
plt.imshow(image, cmap = 'gray')
plt.show()