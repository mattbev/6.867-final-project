import os
import pickle
import torch
import imageio
import itertools
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)


def imshow(img):
    """
    show an image
    Args:
        img (TODO): the image to display
    """    
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.imshow(npimg, cmap='gray')
    plt.show()
    plt.savefig("figures/gan_output.png")
    

def generic_train(model, num_epochs, trainloader, optimizer, criterion, attack, device="cpu", verbose=False):
    """
    train a model
    Args:
        model (torch.nn.Module): the model to train
        num_epochs (int): the number of epochs
        trainloader (torch.utils.data.Dataloader): the training dataset dataloader
        optimizer (torch.optim.*): the function to optimize with
        criterion (torch.nn.*): the loss function
        attack (Attack): the attack model
        device (str or pytorch device, optional): where to evaluate pytorch variables. Defaults to "cpu".
        verbose (bool, optional): extended print statement? Defaults to False.
    Returns:
        (list[float]): the training loss per epoch 
    """ 
    print_every = 50
    if type(device) == str:  
        device = torch.device(device) 
    model = model.to(device)

    # train attack
    attack.train(model, trainloader)

    model.train()
    train_losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple time
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)# get the inputs; data is a list of [inputs, labels]
            inputs, labels = attack.run(inputs, labels)
            inputs.to(device) 
            labels.to(device)
            optimizer.zero_grad() # zero the parameter gradients

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if verbose:
                running_loss += loss.item()
                if i % print_every == 0 and i != 0:  
                    print(f"[epoch: {epoch}, datapoint: {i}] \t loss: {round(running_loss / print_every, 3)}")
                    running_loss = 0.0
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(trainloader)) #this is buggy

    return train_losses
            

def test_total_accurcy(model, testloader, device="cpu"):
    """
    compute the (pure) accuracy over a test set 
    Args:
        model (torch.nn.Module): [description]
        testloader (torch.utils.data.Dataloader): the test set dataloader
        device (str or pytorch device, optional): where to evaluate pytorch variables. Defaults to "cpu".
    Returns:
        (float): the accuracy
    """  
    if type(device) == str:  
        device = torch.device(device) 
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def test_class_accuracy(model, testloader, device="cpu"):
    """
    compute (pure) accuracy per class in the test set
    Args:
        model (torch.nn.Module): the model to evaluate
        testloader (torch.utils.data.Dataloader): the test set dataloader
        device (str or pytorch device, optional): where to evaluate pytorch variables. Defaults to "cpu".
    """    
    if type(device) == str:  
        device = torch.device(device) 
    class_correct = np.array([0. for i in range(10)])
    class_total = np.array([0. for i in range(10)])
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    return class_correct / class_total


def test_confusion_matrix(model, testloader, device="cpu"):
    predictions = torch.Tensor([])
    groundtruths = torch.Tensor([])
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions = torch.cat((predictions, predicted), dim=0)
            groundtruths = torch.cat((groundtruths, labels), dim=0)
    # print(predictions.shape)
    # print(groundtruths.shape)
    # stacked = torch.stack((groundtruths, predictions), dim=1)
    # print(stacked.shape)
    return confusion_matrix(groundtruths, predictions)
    

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def save_model(model, name):
    save_dir = os.path.join("saved_models", name)
    torch.save(model.state_dict(), os.path.join("saved_models", f"{name}.pkl"))


def load_model(model, name):
    model.load_state_dict(torch.load(os.path.join("saved_models", f"{name}.pkl")))
    return model