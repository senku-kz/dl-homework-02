import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda, Compose
from torchvision import datasets, transforms

from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorboard

train_data = './train'
valid_data = './valid'
test_data = './test'


def task_01_a():
    file_paths = pd.read_csv("instruments.csv")
    # print('file_paths:\n', file_paths)

    class_defs = pd.read_csv('class_dict.csv')
    # print('class_defs:\n', class_defs)
    return file_paths, class_defs


def preprocess(folder_path='.'):

    categories = ['train', 'test', 'valid']

    image_size = (224, 224)
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(image_size),
                                                # torchvision.transforms.ToTensor()
                                                ])
    train_d = torchvision.datasets.ImageFolder(root=folder_path + '/' + categories[0], transform=transform)
    test_d = torchvision.datasets.ImageFolder(root=folder_path + '/' + categories[1], transform=transform)
    valid_d = torchvision.datasets.ImageFolder(root=folder_path + '/' + categories[2], transform=transform)
    print(train_d)

    return train_d, test_d, valid_d


def imshow(img):
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def getitem(idx=2):
    img_paths, img_labels = task_01_a()
    img_path = img_paths[img_paths['data set'] == 'train'].iloc[idx, 0]
    img_label = img_paths[img_paths['data set'] == 'train'].iloc[idx, 1]
    print(img_path, img_label)
    # img_path = os.path.join(img_dir, img_labels.iloc[idx, 0])
    image = read_image(img_path)
    # label = self.img_labels.iloc[idx, 1]
    # if self.transform:
    #     image = self.transform(image)
    # if self.target_transform:
    #     label = self.target_transform(label)
    # return image, label


def t1():
    file_paths = pd.read_csv("instruments.csv")
    class_defs = pd.read_csv('class_dict.csv')

    img_path = file_paths[file_paths['data set'] == 'train']
    img_label = file_paths[file_paths['data set'] == 'train']

    print(np.put(file_paths['labels'], class_defs['class'], class_defs['class_index']))
    c = file_paths['labels'].apply(lambda x: class_defs['class'])


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # task_01_a()
    # preprocess()
    # getitem(1)
    t1()
