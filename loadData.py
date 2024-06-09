import argparse
from PIL import Image
import numpy as np
import pandas as pd
from progressbar import *
from torchvision import transforms
from torch.utils.data import dataset, dataloader


class LoadDataset(dataset.Dataset):
    def __init__(self, label_file, num_images, data_dir='data/', transform=None):


    def __getitem__(self, i):


    def __len__(self):
        return self.length


def get_test_trans():
    return transforms.Compose([
        transforms.ToTensor(),

    ])


def get_train_trans():
    return transforms.Compose([
        transforms.ToTensor(),

    ])


def get_dataloader(mode, label_file, opt: argparse.Namespace):
    if mode == 'train':
        return dataloader.DataLoader(
            dataset=LoadDataset(
                label_file=label_file,
                transform=get_train_trans()
            ),
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.tr_dl_num_worker
        )
    elif mode == 'test':
        return dataloader.DataLoader(
            dataset=LoadDataset(
                label_file=label_file,
                transform=get_test_trans()
            ),
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.te_dl_num_worker
        )
    else:
        raise ValueError('Unknown mode: %s' % mode)
