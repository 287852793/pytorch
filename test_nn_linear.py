import torch
import torch.nn as nn
import torchvision.datasets
from torch.nn import Linear
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.linear = Linear(196608, 10)

    def forward(self, input):
        output = self.linear(input)
        return output


dataset = torchvision.datasets.CIFAR10('dataset', train=False, download=True, transform=ToTensor())

dataloader = DataLoader(dataset, 64)

m = M()

for data in dataloader:
    imgs, target = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1, 1, 1, -1))
    output = torch.flatten(imgs)
    print(output.shape)

    output = m(output)
    print(output.shape)
