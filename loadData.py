import argparse
from PIL import Image
import numpy as np
import pandas as pd
from progressbar import *
from torchvision import transforms
from torch.utils.data import dataset, dataloader

widgets = ['Loading images: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]


class LoadDataset(dataset.Dataset):
    def __init__(self, label_file, num_images, data_dir='data/', transform=None):


    def __getitem__(self, i):


    def __len__(self):
        return self.length


def get_test_trans(opt: argparse.Namespace):
    return transforms.Compose([
        transforms.ToTensor(),

    ])


def get_train_trans(opt: argparse.Namespace):
    return transforms.Compose([
        transforms.ToTensor(),

    ])


def get_dataloader(mode, label_file, opt: argparse.Namespace):
    if mode == 'train':
        dataset = LoadDataset(
            label_file=label_file,
            transform=get_train_trans()
        )
        return dataloader.DataLoader(
            dataset=dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.tr_dl_num_worker
        )#, dataset.label_statistics
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
