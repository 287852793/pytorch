import torch
from torch import nn


class MyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output


m = MyModule()
x = torch.tensor(1.0)
output = m(x)
print(output)
