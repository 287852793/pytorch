import torch

first_counter = torch.Tensor([0])
second_counter = torch.Tensor([0])

# print((first_counter < second_counter)[0])

while (first_counter < second_counter):
   first_counter += 2
   second_counter += 1.9

print(first_counter)
print(second_counter)