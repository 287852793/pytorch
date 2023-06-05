from builtins import print

import torch.cuda
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)

trans = transforms.ToTensor()
tensor = trans(img)
# print(tensor)

# print(transforms.ToTensor()(img))

# print(torch.cuda.is_available())

writer = SummaryWriter("logs")
writer.add_image("tensor img", tensor)

writer.close()
