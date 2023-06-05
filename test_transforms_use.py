from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, RandomCrop

img = Image.open("hymenoptera_data/train/ants/0013035.jpg")
print(img)

writer = SummaryWriter("logs")

# tensor
trans = ToTensor()
tensor = trans(img)
writer.add_image("img tensor", tensor)

# normalize
trans2 = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
print(tensor[0][0][0])
tensor2 = trans2(tensor)
print(tensor2[0][0][0])
writer.add_image("img normalize", tensor2)

# resize
trans3 = Resize((512, 512))
print(img.size)
img3 = trans3(img)
print(img3.size)
writer.add_image("img resize", trans(img3))

# compose
trans4 = Compose([Resize((512, 512)), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
tensor4 = trans4(img)
writer.add_image("img compose", tensor4, 1)

# random crop
trans5 = Compose([RandomCrop(50), ToTensor()])
for i in range(10):
    print(i)
    writer.add_image("img random", trans5(img), i)

writer.close()
