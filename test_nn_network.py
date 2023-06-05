import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class M(nn.Module):
    def __init__(self):
        super(M, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.seq(x)
        return x


m = M()
print(m)

dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

loss = nn.CrossEntropyLoss()
optim = torch.optim.SGD(m.parameters(), 0.01)

for epoch in range(20):
    s = 0.0
    for data in dataloader:
        imgs, targets = data
        output = m(imgs)
        res_loss = loss(output, targets)
        optim.zero_grad()
        res_loss.backward()
        optim.step()
        s = s + res_loss
    print(s)

# writer = SummaryWriter("logs")
# writer.close()
