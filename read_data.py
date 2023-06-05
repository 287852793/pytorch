import os

from caffe2.python.dataset import Dataset
from torch.utils.data import Dataset
from PIL import Image


class MyData(Dataset):

    def __init__(self, root, label):
        self.root = root
        self.label = label
        self.path = os.path.join(root, label)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root, self.label, img_name)
        img = Image.open(img_item_path)
        label = self.label
        # hymenoptera_data / train / ants / 0013035.jpg

    def __len__(self):
        return len(self.img_path)


root = 'hymenoptera_data/train'
ants_label = 'ants'
ants_dataset = MyData(root, ants_label)

bees_label = 'bees'
bees_dataset = MyData(root, bees_label)

train_dataset = ants_dataset + bees_dataset

