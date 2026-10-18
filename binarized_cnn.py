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

model = CNN().to(device)

optimizer = optim.Adam(model.parameters(), lr = 0.001)

criterion = nn.CrossEntropyLoss()

# Display a binarised number
image, label = train_data[1]

print("Shape:", image.shape)
print("Unique values:", torch.unique(image))
print("Label:", label)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()