# -*- coding: utf-8 -*-

#########################################################################
##                        Libraries                                    ##
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#Importing Libraries 
from tqdm import tqdm
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


#Importing custom packages
from models import FashionMNISTCNN


#Server Aggregation
def aggregate(global_model, client_models):
    """
    global parameter updates aggregation.

    Args:
        global_model (torch.nn.Module): the global model
        client_models (list[torch.nn.Module]): the client models
    """    
    ### take simple mean of the weights of models ###
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        global_dict[k] = torch.stack([client_models[i].state_dict()[k].float() for i in range(len(client_models))], 0).mean(0)
    global_model.load_state_dict(global_dict)
    for model in client_models:
        model.load_state_dict(global_model.state_dict())
          
          
#########################################################################
##                    Parameters                                       ##
#########################################################################
    
device = torch.device("cuda:0") #GPU
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 2
m = 10     #number of participants
Rounds =5
 
#########################################################################
##                    Importing Dataset                                ##
#########################################################################

#transforming the PIL Image to tensors and normalize to [-1,1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, ), (0.5, ),)])



trainset = torchvision.datasets.FashionMNIST(root = "./data", train = True, download = True, transform = transform)
testset = torchvision.datasets.FashionMNIST(root = "./data", train = False, download = True, transform = transform)

#Randomly splitting the dataset 
trainset_split = torch.utils.data.random_split(trainset, [int(trainset.data.shape[0] / m) for _ in range(m)])

# Creating a pytorch loader for a Deep Learning model
trainloader = [torch.utils.data.DataLoader(x, batch_size=BATCH_SIZE, shuffle=True) for x in trainset_split]

#loading the test data from testset
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

#Classes name
classes = ('T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot')


########################################################################
##                   Neural Network Architecture                      ##
########################################################################

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

#Define global and client models

global_model =  FashionMNISTCNN

############## client models ##############
client_models = [ FashionMNISTCNN  for _ in range(m)]
for model in client_models:
    model.load_state_dict(global_model.state_dict()) ### initial synchronizing with global model 

############### optimizers ################
opt = [optim.SGD(FashionMNISTCNN.parameters(), lr=LEARNING_RATE, momentum=0.9) for model in client_models]

###### Federated Learning  ################
losses_train = []
acc_train = []
# Runnining FL

for r in range(Rounds):
    # select random clients
    client_idx = np.random.permutation(m)[:m]
    # client update
    loss = 0
    for i in range(m):
         
       loss += train( client_models[i], device, trainloader[client_idx[i]],opt[i], NUM_EPOCHS)
       print(' client %i  successfully trained the model' % i)
    losses_train.append(loss)
    # server aggregate
    aggregate(global_model, client_models)
    print('%d-th round' % r)
    print('average train loss %0.3g ' % (loss / m))
    


########################################################################
##                        Testing                                     ##
########################################################################

#Overall Accuracy

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = global_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))





