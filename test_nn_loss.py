import torch

inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss_l1 = torch.nn.L1Loss(reduction='sum')
res = loss_l1(inputs, targets)

print(res)

loss_mse = torch.nn.MSELoss()
res = loss_mse(inputs, targets)

print(res)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
loss_cross = torch.nn.CrossEntropyLoss()
res = loss_cross(x, y)

print(res)
