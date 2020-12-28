import torch
from torch.utils import data
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

import numpy as np
import os, psutil
import glob
from bisect import bisect

class MnistDataset(data.Dataset):
    def __init__(self, npy_files_path, num_classes, transforms=False):
        npy_images_files = glob.glob(os.path.join(npy_files_path, "*images*.npy"))
        npy_labels_files = glob.glob(os.path.join(npy_files_path, "*labels*.npy"))
        
        npy_images_files.sort()
        npy_labels_files.sort()
        
        self.image_memmaps = [np.load(npy_images_file, mmap_mode="r") for npy_images_file in npy_images_files]
        self.label_memmaps = [np.load(npy_labels_file, mmap_mode="r") for npy_labels_file in npy_labels_files]
        
        self.start_indices = [0] * len(npy_labels_files)
        self.data_count = 0
        for index, memmap in enumerate(self.image_memmaps):
            self.start_indices[index] = self.data_count
            self.data_count += memmap.shape[0]
        
        self.num_classes = num_classes
        self.transforms = transforms

    def __len__(self):
        return self.data_count

    def __getitem__(self, index):
        memmap_index = bisect(self.start_indices, index) - 1
        index_in_memmap = index - self.start_indices[memmap_index]
        
        data = torch.from_numpy(np.array(self.image_memmaps[memmap_index][index_in_memmap]))
        label = torch.from_numpy(np.array(self.label_memmaps[memmap_index][index_in_memmap]))

        return torch.unsqueeze(data, 0), label

    def __del__(self):
        del self.image_memmaps
        del self.label_memmaps

if __name__ == "__main__":
    process = psutil.Process(os.getpid())
    memory_before = process.memory_info().rss
    dataset = MnistDataset("../../../data/mnist/train", 10)
    
    used_memory = process.memory_info().rss - memory_before
    print(used_memory)
    image, label = dataset[3]
    print(image.shape)
    