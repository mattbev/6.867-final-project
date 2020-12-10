import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt  

def imshow(img):
    """
    show an image
    Args:
        img (TODO): the image to display
    """    
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    

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