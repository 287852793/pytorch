import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


m = M2()
print(m)

writer = SummaryWriter("logs")

i = 0
for data in dataloader:
    imgs, targets = data
    output = m(imgs)
    # print(imgs.shape)
    # print(output.shape)

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("input", imgs, i)
    writer.add_images("output", output, i)

    i += 1

writer.close()
