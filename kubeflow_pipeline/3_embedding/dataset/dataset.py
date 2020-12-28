import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

import numpy as np
import os, psutil
import glob
from bisect import bisect

class EmbedDataset(data.Dataset):
    def __init__(self, npy_files_path):
        npy_images_files = glob.glob(os.path.join(npy_files_path, "*images*.npy"))
        npy_images_files.sort()
        
        self.image_memmaps = [np.load(npy_images_file, mmap_mode="r") for npy_images_file in npy_images_files]
        
        self.start_indices = [0] * len(npy_images_files)
        self.data_count = 0
        for index, memmap in enumerate(self.image_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        
        data = torch.from_numpy(np.array(self.image_memmaps[memmap_index][index_in_memmap]))

        return torch.unsqueeze(data, 0)
    
    def __del__(self):
        del self.image_memmaps