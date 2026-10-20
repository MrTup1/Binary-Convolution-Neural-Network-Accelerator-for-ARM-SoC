from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
matplotlib.use("qtagg")
import matplotlib.pyplot as plt

# Binarize image so that every pixel becomes either 1.0 or -1.0
binary_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda( 
        lambda x: torch.where(x >= 128 / 255, torch.tensor(1.0), torch.tensor(-1.0))
    )
])

train_data = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=binary_transform
)

test_data = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=binary_transform
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

def n_ary(w, sigma, delta):
    return 0.5 * (
        torch.tanh(sigma * (w + delta)) +
        torch.tanh(sigma * (w - delta))
    )

class NaryConv2d(nn.Conv2d):
    def __init__(self, *args, delta = 0.0, **kwargs): # Pass arguments by position and name
        super().__init__(*args, **kwargs)
        self.delta = delta

        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, sigma):
        weights_nary = self.a * n_ary(self.weight, sigma, self.delta)
        x = F.conv2d(x, weights_nary, self.bias, stride = self.stride, padding = self.padding, dilation = self.dilation, groups = self.groups)

        return x
    
class NaryDense(nn.Linear):
    def __init__(self, *args, delta = 0.0, **kwargs): # Pass arguments by position and name
        super().__init__(*args, **kwargs)
        self.delta = delta

        self.a = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, sigma):
        weights_nary = self.a * n_ary(self.weight, sigma, self.delta)

        return F.linear(x, weights_nary, self.bias)
    
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigma = 1.0

        self.conv1 = NaryConv2d(1, 16, kernel_size = 3, stride = 1, delta = 0.0)
        self.max1 = nn.MaxPool2d(kernel_size = 3, stride = 3)
        self.fc1 = NaryDense(1024, 32, delta = 0.0)
        self.fc2 = NaryDense(32, 10, delta = 0.0)

    def forward(self, x):
        x = self.conv1(x, self.sigma)

        x = n_ary(x, self.sigma, 0.0)
        x = self.max1(x)
        x = x.view(x.size(0), -1) #Flattens all dimensions except batch size (index 0) for Linear Layer
        x = self.fc1(x, self.sigma)
        x = n_ary(x, self.sigma, 0.0)
        x = self.fc2(x, self.sigma)

        return x
    
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(0)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)

criterion = nn.CrossEntropyLoss()

schedule = [
    (1, 3),
    (2, 3),
    (4, 3),
    (8, 3),
    (16, 3),
    (32, 3),
    (64, 3),
    (128, 3),
    (256, 3),
    (512, 3),
    (768, 3),
    (1024, 1)
] #Page 205 Drupelet

def train():
    model.train()

    running_loss = 0

    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()


    return (running_loss / len(loaders['train'])) # return average loss in this epoch 

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


for sigma, epochs in schedule:
    model.sigma = sigma 

    print(f"Sigma: {model.sigma}")

    for epoch in range(epochs):
        train_loss = train()

        print(f"Train Loss: {train_loss:.4f}")

test()


print("Final sigma:", model.sigma)

with torch.no_grad():
    for name, layer in [
        ("conv1", model.conv1),
        ("fc1", model.fc1),
        ("fc2", model.fc2)
    ]:
        g = n_ary(
            layer.weight,
            model.sigma,
            layer.delta
        )

        fraction = (
            (g.abs() > 0.999)
            .float()
            .mean()
            .item()
        )

        print(
            name,
            "binary fraction:",
            fraction,
            "a:",
            layer.a.item()
        )

# Display a binarised number
model.eval()
data, target = test_data[1]
data = data.unsqueeze(0).to(device)
output = model(data)
prediction = output.argmax(dim = 1, keepdim = True).item()
print(f"Prediction: {prediction}")
image = data.squeeze(0).squeeze(0).cpu().numpy()
plt.imshow(image, cmap = 'gray')
plt.show()