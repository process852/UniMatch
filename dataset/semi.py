from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size

        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None:
                self.ids *= math.ceil(nsample / len(self.ids))
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        ## use for change detection, we use three images, A, B, Label
        imgA = Image.open(id.split(' ')[0]).convert("RGB")
        imgB = Image.open(id.split(' ')[1]).convert("RGB")
        mask = Image.fromarray(np.array(Image.open(id.split(' ')[2])))
        # img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        # mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1]))))

        if self.mode == 'val':
            imgA, imgB, mask = normalize(imgA, imgB, mask)
            return imgA, imgB, mask, id

        imgA, imgB, mask = resize(imgA, imgB, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        imgA, imgB, mask = crop(imgA, imgB, mask, self.size, ignore_value)
        imgA, imgB, mask = hflip(imgA, imgB, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(imgA, imgB, mask)

        imgA_w, imgA_s1, imgA_s2 = deepcopy(imgA), deepcopy(imgA), deepcopy(imgA)
        imgB_w, imgB_s1, imgB_s2 = deepcopy(imgB), deepcopy(imgB), deepcopy(imgB)

        if random.random() < 0.8:
            imgA_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s1)
            imgB_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgB_s1)
        # img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        imgA_s1, imgB_s1= blur(imgA_s1, imgB_s1, p=0.5)
        # cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            imgA_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgA_s2)
            imgB_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(imgB_s2)
        # img_s2 = transforms.RandomGrayscale(p=0.2)(imgA_s2)
        imgA_s2, imgB_s2 = blur(imgA_s2, imgB_s2, p=0.5)
        # cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        imgA_s1, imgB_s1 = normalize(imgA_s1, imgB_s1)
        imgA_s2, imgB_s2 = normalize(imgA_s2, imgB_s2)

        mask = np.array(mask)
        mask[mask > 127] = 1
        mask = torch.from_numpy(mask).long()
        # ignore_mask[mask == 254] = 255
        imgA_w, imgB_w = normalize(imgA_w, imgB_w)
        return imgA_w, imgB_w, imgA_s1, imgB_s1, imgA_s2, imgB_s2, mask

    def __len__(self):
        return len(self.ids)
