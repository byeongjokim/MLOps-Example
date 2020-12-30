import torch
from torch.utils import data

import cv2
import numpy as np
import os
import glob

class MnistDataset(data.Dataset):
    def __init__(self, dataset_folder, num_classes, shape=(28,28), transforms=False):
        self.image_files = glob.glob(os.path.join(dataset_folder, "**/*.png"))
        self.labels = [int(i.split(os.sep)[-2]) for i in self.image_files]

        self.num_classes = num_classes
        self.shape = shape
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.shape)

        image = torch.from_numpy(image).unsqueeze(0)

        return image.float(), self.labels[index]