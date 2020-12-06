# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:24:35 2020

@author: victo
"""
#########################################################################
##                        Libraries                                    ##
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#Importing Libraries 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import numpy as np

#########################################################################
##                    Auxiliary functions                              ##
#########################################################################


#Display image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    


#Training
def train( model, device, train_loader,optimizer, NUM_EPOCHS):
    model.train()
    for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # print statistics
            running_loss += loss.item()
            if i % 32 == 31:    # print every 2000 mini-batches
               print('[%d, %5d] loss: %.3f' %
                     (epoch + 1, i + 1, running_loss / 2000))
               running_loss = 0.0
            
            


#########################################################################
##                    Parameters                                       ##
#########################################################################
    
device = torch.device("cuda:0") #GPU
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 2

#########################################################################
##                    Importing Dataset                                ##
#########################################################################

#transforming the PIL Image to tensors and normalize to [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ),)])



trainset = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transform)
testset = torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = transform)


#loading the training data from trainset
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle = True)
#loading the test data from testset
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

#Classes name
classes = ('T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot')


########################################################################
##                   Neural Network Architecture                      ##
########################################################################

class FashionMNISTCNN(nn.Module):

    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))

        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

FashionMNISTCNN = FashionMNISTCNN()
#Run on GPU
FashionMNISTCNN = FashionMNISTCNN.to(device)
########################################################################
##                         Loss Function                              ##
########################################################################

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(FashionMNISTCNN.parameters(), lr=LEARNING_RATE, momentum=0.9)


########################################################################
##                        Training                                    ##
########################################################################

#Train on GPU
train( FashionMNISTCNN, device, trainloader,optimizer, NUM_EPOCHS)

########################################################################
##                        Testing                                     ##
########################################################################

#Overall Accuracy

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = FashionMNISTCNN(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


#For each class

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = FashionMNISTCNN(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))


