import random
import time
from datetime import datetime

import matplotlib.image as img
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from MIDataset import MIDataset

epochs = 20
batch_size = 8
prod = True


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
    # epochs = 20
    # batch_size = 16
    learning_rate = 0.01

    train, test, valid = preprocess()
    train_size = len(train) if prod else 128

    transform = transforms.Compose([
        transforms.ToPILImage('RGB'),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ===============Train
    print('Epochs:%d \t Train size: %d \t Batch size: %d' % (epochs, train_size, batch_size,))
    train_ds = MIDataset(train_size, 0, transform=transform)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )

    test_ds = MIDataset(len(test), 1, transform=transform)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, )

    model_trained = googlenet_train(train_dataloader, learning_rate, batch_size, epochs)
    print('=' * 50)
    model_accuracy = googlenet_test(model_trained, test_dataloader)
    print(model_accuracy)
    save_model(model_trained, train_ds, 'GoogLeNet', model_accuracy[0])


def googlenet_train(dataloader, learning_rate=0.01, batch_size=16, epochs=20):
    mini_batch = 10
    googLeNet_model = torchvision.models.googlenet(init_weights=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    googLeNet_model.to(device)

    # criterion = nn.MSELoss()  # squared error
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(googLeNet_model.parameters(), lr=learning_rate, momentum=0.9)

    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(epoch + 1, '=' * 50)
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = googLeNet_model(inputs)
            # print('=======', outputs[0].shape, labels.shape, '======')
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
            #
            running_loss += loss.item()
            if i % mini_batch == mini_batch - 1:  # print every batches
                print('\t[epoch %2d, i %3d] loss: %.3f' % (epoch + 1, i + 1, round(running_loss / batch_size, 2)))
                running_loss = 0.0

            # if i % 3 == 0:
            #     print('Loss:', round(loss.item(), 2), 'From:', (i + 1) * batch_size)

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
            # print('label:', labels[0], 'predicted', predicted[0])

    accuracy_percent = 100 * correct / total
    print('Accuracy is %d%%, the number of correct/total: %d/%d' % (accuracy_percent, correct, total))
    return accuracy_percent, correct, total


def save_model(model, dataset, name='CNN', accuracy=''):
    v_date = datetime.now().strftime("%Y%m%d-%H%M")
    file_name_1 = '{}-Accuracy-{}-{}-{}.model'.format(v_date, round(accuracy, 2), name, len(dataset))
    file_name_2 = '{}-Accuracy-{}-{}-{}.param'.format(v_date, round(accuracy, 2), name, len(dataset))
    torch.save(model.state_dict(), file_name_1)
    torch.save(model, file_name_2)


def googlenet_train_with_accuracy(dataloader_train, dataloader_test, learning_rate=0.01, batch_size=16, epochs=20):
    mini_batch = 10
    accuracies = []
    googLeNet_model = torchvision.models.googlenet(init_weights=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    googLeNet_model.to(device)

    # criterion = nn.MSELoss()  # squared error
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(googLeNet_model.parameters(), lr=learning_rate, momentum=0.9)

    start = time.time()
    for epoch in range(epochs):  # loop over the dataset multiple times
        print(epoch + 1, '=' * 50)
        running_loss = 0.0
        for i, data in enumerate(dataloader_train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = googLeNet_model(inputs)
            loss = criterion(outputs[0], labels)
            loss.backward()
            optimizer.step()
            #
            running_loss += loss.item()
            if i % mini_batch == mini_batch - 1:  # print every batches
                print('\t[epoch %2d, i %5d] loss: %.3f' % (epoch + 1, i + 1, round(running_loss / batch_size, 2)))
                running_loss = 0.0

        accuracy = round(googlenet_test(googLeNet_model, dataloader_test)[0], 2)
        accuracies.append(accuracy)
    print('Finished Training')
    end = time.time()
    print('Training time:', end - start)
    return accuracies


def task_03():
    """
    Plot the graph showing the accuracy on a test set with different values of learning rate.
    Learning rate values should be 0.1, 0.01, 0.001.
    """
    # epochs = 20
    # batch_size = 16
    learning_rates = [0.1, 0.01, 0.001]

    train, test, valid = preprocess()
    train_size = len(train) if prod else 128

    transform = transforms.Compose([
        transforms.ToPILImage('RGB'),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print('Epochs:%d \t Train size: %d \t Batch size: %d' % (epochs, train_size, batch_size,))
    train_ds = MIDataset(train_size, 0, transform=transform)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )

    test_ds = MIDataset(len(test), 1, transform=transform)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, )
    lst_accuracy = []
    for learning_rate in learning_rates:
        accuracy_model = googlenet_train_with_accuracy(train_dataloader, test_dataloader, learning_rate=learning_rate,
                                                       batch_size=batch_size, epochs=epochs)
        lst_accuracy.append(accuracy_model)

    x_epochs = range(1, epochs + 1)
    fig = plt.figure()
    fig.set_size_inches(10, 10)
    plt.plot(x_epochs, lst_accuracy[0], 'r', label='Learning rate = ' + learning_rates[0].__str__())
    plt.plot(x_epochs, lst_accuracy[1], 'g', label='Learning rate = ' + learning_rates[1].__str__())
    plt.plot(x_epochs, lst_accuracy[2], 'b', label='Learning rate = ' + learning_rates[2].__str__())
    plt.title('Effect of various learning rates')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy, %')
    plt.legend()
    plt.show()


def three_models_train(dataloader, learning_rate=0.01, batch_size=16, epochs=20):
    alexNet_model = torchvision.models.AlexNet()
    resnet_model = torchvision.models.resnet18()
    denseNet_model = torchvision.models.densenet121()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    models = [alexNet_model, resnet_model, denseNet_model]
    models_trained = []

    mini_batch = 10
    for model in models:
        # criterion = nn.MSELoss()  # squared error
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        start = time.time()
        for epoch in range(epochs):  # loop over the dataset multiple times
            print(epoch + 1, '=' * 50)
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                torch.cuda.empty_cache()
                del inputs, labels

                running_loss += loss.item()
                if i % mini_batch == mini_batch - 1:  # print every batches
                    print('\t[epoch %2d, i %3d] loss: %.3f' % (epoch + 1, i + 1, round(running_loss / batch_size, 2)))
                    running_loss = 0.0

        print('Finished Training')
        end = time.time()
        print('Training time:', end - start)
        models_trained.append(model)
    return models_trained


def three_models_test(models, dataloader):
    accuracy = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for t_model in models:
        t_model.to(device)
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = t_model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print('label:', labels[0], 'predicted', predicted[0])

        accuracy_percent = 100 * correct / total
        print('Accuracy is %d%%, the number of correct/total: %d/%d' % (accuracy_percent, correct, total))
        accuracy.append((accuracy_percent, correct, total,))
    return accuracy


def task_04():
    """
    Choose any 3 variants of CNN architecture and compare their accuracy.
    """
    # epochs = 10
    # batch_size = 8
    learning_rates = 0.001

    train, test, valid = preprocess()
    train_size = len(train) if prod else 128

    transform = transforms.Compose([
        transforms.ToPILImage('RGB'),
        # transforms.ToPILImage('F'),
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print('Epochs:%d \t Train size: %d \t Batch size: %d' % (epochs, train_size, batch_size,))
    train_ds = MIDataset(train_size, 0, transform=transform)
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )

    test_ds = MIDataset(len(test), 1, transform=transform)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, )
    lst_accuracy = []
    # print(len(train_dataloader))
    models_trained = three_models_train(train_dataloader, learning_rates, batch_size, epochs)
    print('=' * 50)
    models_accuracy = three_models_test(models_trained, test_dataloader)
    # print(models_accuracy)

    print('*' * 50)
    print('\tAlexNet accuracy (%%):\t%d' % round(models_accuracy[0][0], 2))
    print('\tResNet accuracy (%%):\t%d' % round(models_accuracy[1][0], 2))
    print('\tDenseNet accuracy (%%):\t%d' % round(models_accuracy[2][0], 2))
    print('*' * 50)


if __name__ == "__main__":
    # task_01_a()
    # task_01_b()
    # task_01_c()
    # task_02()
    # task_03()
    task_04()
