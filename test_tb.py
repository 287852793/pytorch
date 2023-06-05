from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
img = Image.open("hymenoptera_data/train/ants/0013035.jpg")
arr = np.array(img)
writer.add_image('test', arr, 1)
# writer.add_image('test', arr, 1, dataformats="HWC")
# writer.add_scalar()

# for i in range(100):
#     writer.add_scalar("y=2x", 3 * i, i)

writer.close()
