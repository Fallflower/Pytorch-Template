# 从硬盘加载数据，并预处理

import numpy as np
# import torchvision.transforms
from torchvision import transforms
from torch.utils.data import dataset

import os


def load_data(root, file_name, label_file_name):
    # 
    return x, y, length


class LoadDataset(dataset.Dataset):
    def __init__(self, root, data_name, label_name, transform=None):
        xs, ys, length = load_data(root, data_name, label_name)
        self.x = xs
        self.y = ys
        self.length = length
        self.transform = transform

    def __getitem__(self, i):
        img, target = self.x[i].copy(), self.y[i]
        # print(img.shape)
        # print((target))
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return self.length


trans = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

trainDataset = LoadDataset(
    'data/',
    # "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
    transform=trans
)
testDataset = LoadDataset(
    'data/',
    # "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz",
    transform=trans
)
