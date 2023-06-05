import torchvision

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

train_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True)

print(test_dataset[0])