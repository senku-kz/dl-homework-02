import random
import torch
import torchvision

import numpy as np
from torch.utils.data import Dataset

import PIL.Image as Image


class MIDataset(Dataset):
    def __init__(self, dataset_size, dataset_type, transform=None, target_transform=None):
        self.dataset_size = dataset_size
        self.dataset_type = dataset_type
        # self.labels = np.arange(dataset_size)  # note that labels are integer values in range [0,29]
        # extract the images f
        # self.images =
        self.transform = transform
        self.target_transform = target_transform
        # PUT YOU CODE HERE
        from homework import preprocess
        ds = preprocess()[dataset_type]
        lst_images = []
        lst_labels = []
        for i in range(dataset_size):
            random_index = random.randrange(len(ds))
            img = torchvision.io.read_image(ds.iloc[random_index][0], torchvision.io.ImageReadMode.RGB)
            # img = Image.open(ds.iloc[random_index][0]).convert('RGB')
            # print(img.shape)
            # lst_images.append(ds.iloc[random_index][0])
            lst_images.append(img)
            lst_labels.append(ds.iloc[random_index][1])

        self.images = lst_images
        self.labels = np.array(lst_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        X = self.images[idx]
        Y = self.labels[idx]
        if self.transform:
            X = self.transform(X)
        return X, Y
