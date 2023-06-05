import torch

import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)

input = torch.reshape(input, [-1, 1, 5, 5])

print(input.shape)

test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,
                                            transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(test_dataset, batch_size=64)


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.maxpool = nn.MaxPool2d(3, ceil_mode=False)

    def forward(self, x):
        r = self.maxpool(x)
        return r


m = M()
output = m(input)
print(output)

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("test", imgs, step)
    output = m(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()
