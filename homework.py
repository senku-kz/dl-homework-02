import random
import time
from datetime import datetime

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
import matplotlib.image as img
import tensorboard

from MIDataset import MIDataset


# train_data = './train'
# valid_data = './valid'
# test_data = './test'


def task_01_a():
    """
    Complete a preprocess function to read the dataset into train, test and validation sets.
    Use instruments.csv and class_dict.csv to find a filepath, class label and index information.
    """
    file_paths = pd.read_csv("instruments.csv")
    # print('file_paths:\n', file_paths)

    class_defs = pd.read_csv('class_dict.csv')
    # print('class_defs:\n', class_defs)

    train, test, valid = preprocess()
    print(train)
    print(test)
    print(valid)


def task_01_b():
    """Plot 25 random sample of images from the train folder."""
    train, _, _ = preprocess()
    class_label = get_class()

    rows = 5
    columns = 5
    img_num = 1

    fig = plt.figure()
    fig.set_size_inches(15, 15)
    for r in range(1, rows + 1):
        for c in range(1, columns + 1):
            random_index = random.randrange(len(train))
            img_url, img_label_idx = train.iloc[random_index][0], train.iloc[random_index][1]

            # reading images
            image = img.imread(img_url)

            # Adds a subplot at the 1st position
            fig.add_subplot(rows, columns, img_num)
            fig.tight_layout()
            plt.imshow(image)
            plt.title(class_label[img_label_idx])
            plt.axis('off')
            img_num += 1
    plt.show()


def get_class():
    class_defs = pd.read_csv('class_dict.csv')
    return {class_defs.iloc[i][0]: class_defs.iloc[i][1] for i in range(len(class_defs))}


def preprocess(folder_path='.'):
    dataset = {}
    categories = ['train', 'test', 'valid']
    file_paths = pd.read_csv("instruments.csv")
    class_defs = pd.read_csv('class_dict.csv')

    for category in categories:
        v_file_paths = file_paths[file_paths['data set'] == category]
        r = pd.merge(
            v_file_paths,
            class_defs,
            how="inner",
            left_on='labels',
            right_on='class',
        )
        dataset[category] = r.iloc[:, [0, 3]]

    train_d, test_d, valid_d = dataset['train'], dataset['test'], dataset['valid']
    return train_d, test_d, valid_d


def task_01_c():
    """
    Define a MIDataset class to represent musical instruments dataset.
    It should have a:
        dataset_size parameter that returns a dataset containing dataset_size random sample of images
        dataset_type parameter that can be 0,1 or 2 for train, test, validation splits respectively
    Use a preprocess function to help you with loading data in the dataset class.
    """
    dataset_size = random.randrange(200)
    random_index = random.randrange(dataset_size)

    obj = MIDataset(dataset_size, 0)
    print('len', dataset_size, obj.__len__())
    print('idx', random_index, obj.__getitem__(random_index))


def task_02():
    """
    Train any CNN model
    """
    epochs = 20
    batch_size = 16
    learning_rate = 0.01

    writer = SummaryWriter(log_dir='logs/googlenet')

    train, test, valid = preprocess()
    transform = transforms.Compose([
        transforms.ToPILImage('RGB'),
        # transforms.ToPILImage('F'),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ===============Train
    # train_ds = MIDataset(32, 0, transform=transform)
    train_ds = MIDataset(len(train), 0, transform=transform)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )

    test_ds = MIDataset(len(test), 1, transform=transform)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, )

    # for X, y in train_dataloader:
    #     print("Shape of X [N, C, H, W]: ", X.shape)
    #     print("Shape of y: ", y.shape, y.dtype)

    # resnet18 = torchvision.models.resnet18()
    # alexnet = torchvision.models.alexnet()
    # vgg19 = torchvision.models.vgg19_bn()
    googLeNet_model = torchvision.models.googlenet(init_weights=True)
    # googLeNet_model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    googLeNet_model.to(device)

    # criterion = nn.MSELoss()  # squared error
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(googLeNet_model.parameters(), lr=learning_rate, momentum=0.9)

    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(epoch, '=' * 50)
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = googLeNet_model(inputs)
            # print(outputs[0])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print('=======', outputs[0].shape, labels.shape, '======')
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
            #
            if i % 3 == 0:
                print('Loss:', round(loss.item(), 2), 'From:', (i + 1) * batch_size)

    print('Finished Training')
    end = time.time()
    print('Training time:', end - start)

    # ===============Test
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = googLeNet_model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs[0], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('label:', labels[0], 'predicted', predicted[0])

    accuracy_total = 100 * correct / total
    print('Accuracy is %d%%, the number of correct/total: %d/%d' % (accuracy_total, correct, total))

    save_model(googLeNet_model, train_ds, 'GoogLeNet', accuracy_total)


def googlenet_train(dataloader, learning_rate=0.01, batch_size=16, epochs=20):
    googLeNet_model = torchvision.models.googlenet(init_weights=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    googLeNet_model.to(device)

    # criterion = nn.MSELoss()  # squared error
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(googLeNet_model.parameters(), lr=learning_rate, momentum=0.9)

    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(epoch, '=' * 50)
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = googLeNet_model(inputs)
            # print(outputs[0])

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # print('=======', outputs[0].shape, labels.shape, '======')
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
            #
            if i % 3 == 0:
                print('Loss:', round(loss.item(), 2), 'From:', (i + 1) * batch_size)

    print('Finished Training')
    end = time.time()
    print('Training time:', end - start)
    return googLeNet_model


def googlenet_test(googlenet_model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    googlenet_model.to(device)
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = googlenet_model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs[0], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print('label:', labels[0], 'predicted', predicted[0])

    accuracy_total = 100 * correct / total
    print('Accuracy is %d%%, the number of correct/total: %d/%d' % (accuracy_total, correct, total))
    return accuracy_total, correct, total


def save_model(model, dataset, name='CNN', accuracy=''):
    v_date = datetime.now().strftime("%Y%m%d-%H%M")
    file_name_1 = '{}-Accuracy-{}-{}-{}.model'.format(v_date, round(accuracy, 2), name, len(dataset))
    file_name_2 = '{}-Accuracy-{}-{}-{}.param'.format(v_date, round(accuracy, 2), name, len(dataset))
    torch.save(model.state_dict(), file_name_1)
    torch.save(model, file_name_2)


def task_03():
    """
    Plot the graph showing the accuracy on a test set with different values of learning rate.
    Learning rate values should be 0.1, 0.01, 0.001.
    """
    epochs = 20
    batch_size = 16
    learning_rates = [0.1, 0.01, 0.001]

    train, test, valid = preprocess()
    transform = transforms.Compose([
        transforms.ToPILImage('RGB'),
        # transforms.ToPILImage('F'),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = MIDataset(len(train), 0, transform=transform)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )

    test_ds = MIDataset(len(test), 1, transform=transform)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, )
    lst_accuracy = []
    for learning_rate in learning_rates:
        learned_model = googlenet_train(train_ds, learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)
        accuracy_model = googlenet_test(learned_model, test_ds)
        # print(accuracy_model)
        save_model(learned_model, train_ds, 'GoogLeNet-'+learning_rate, accuracy_model[0])

        lst_accuracy.append(accuracy_model)
    print(lst_accuracy)


if __name__ == "__main__":
    # task_01_a()
    # task_01_b()
    # task_01_c()
    task_02()
