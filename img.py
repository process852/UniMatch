from PIL import Image
import numpy as np


mask = Image.fromarray(np.array(Image.open("/data/home/jinjuncan/dataset/LEVIR_CD/val/label/val_1_2.png")))
print(np.array(mask).max())