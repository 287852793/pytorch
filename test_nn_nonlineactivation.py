import torch
import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

input = torch.tensor([[1, -3],
                      [2, -7]])

data = torch.reshape(input, [-1, 1, 2, 2])

print(data.shape)

dataset = torchvision.datasets.CIFAR10("dataset", transform=ToTensor(), download=True, train=False)
dataloader = DataLoader(dataset, 64)


class M(torch.nn.Module):
    def __init__(self):
        super(M, self).__init__()
        # self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input):
        # output = self.relu(input)
        output = self.sigmoid(input)
        return output


m = M()
res = m(data)
print(res)

writer = SummaryWriter("logs")

step = 0
for i in dataloader:
    imgs, targets = i
    out = m(imgs)
    writer.add_images("act", out, step)
    step += 1

writer.close()
