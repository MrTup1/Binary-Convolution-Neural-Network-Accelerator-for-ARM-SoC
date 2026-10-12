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

# Display a binarised number
image, label = train_data[1]

print("Shape:", image.shape)
print("Unique values:", torch.unique(image))
print("Label:", label)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()