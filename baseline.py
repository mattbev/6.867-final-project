# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from copy import deepcopy

from models import FashionMNISTCNN
from utils import generic_train, test_total_accurcy, test_class_accuracy
torch.manual_seed(1) #Set seed 

class Baseline:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = FashionMNISTCNN()
        self.model.to(self.device)

    def load_data(self, batch_size=32):
        """
        load FashionMNIST data
        Args:
            batch_size (int, optional): the batch size. Defaults to 32.
        """        
        self.batch_size = batch_size
        #transforming the PIL Image to tensors and normalize to [-1,1]
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ),)])
        self.trainset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        self.testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)
        
        self.classes = ('T-Shirt','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot')
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=batch_size, shuffle = True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=batch_size, shuffle=False)

    def test(self):
        """
        test the accuracy of the model
        Returns:
            (tuple[float]): the overall and class-wise accuracies of the model
        """        
        total_acc = test_total_accurcy(self.model, self.testloader, self.device)
        class_acc = test_class_accuracy(self.model, self.testloader, self.device)
        return total_acc, class_acc

    def _make_optimizer_and_loss(self, lr, momentum=0.9):
        """
        helper function to create an optimizer and loss function
        Args:
            lr (float): the learning rate
            momentum (float, optional): the momentum. Defaults to 0.9.
        Returns:
            (tuple[torch.nn.CrossEntropyLoss, torch.optim.SGD]): criterion and optimizer functions
        """        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        return criterion, optimizer



class BasicBaseline(Baseline):
    def __init__(self, device="cpu"):
        super(BasicBaseline, self).__init__(device=device)
        
    def set_trainloader(self, trainloader):
        """
        set the trainloader for the model
        Args:
            trainloader (torch.utils.data.Dataloader): the training data data loader
        """        
        self.trainloader = trainloader

    def train(self, num_epochs, malicious,lr=1e-3, verbose=False):
        """
        train the basic baseline model
        Args:
            num_epochs (int): the number of epochs
            lr (float, optional): the learning rate. Defaults to 1e-3.
            verbose (bool, optional): do you want print output? Defaults to False.
        Returns:
            (list[floats]): the training losses
        """        
        criterion, optimizer = self._make_optimizer_and_loss(lr)
        return generic_train(
            model=self.model, 
            num_epochs=num_epochs, 
            trainloader=self.trainloader, 
            optimizer=optimizer, 
            criterion=criterion,  
            malicious = malicious,
            tl = tl,
            tc = tc,
            device=self.device, 
            verbose=verbose)



class FederatedBaseline(Baseline):
    def __init__(self, num_clients, device="cpu"):
        super(FederatedBaseline, self).__init__(device=device)
        self.num_clients = num_clients

    def train(self, num_epochs, rounds, lr=1e-3, verbose=False):
        """
        train the federated baseline model
        Args:
            num_epochs (int): the number of epochs
            rounts (int): the number of rounds to train clients
            lr (float, optional): the learning rate. Defaults to 1e-3.
            verbose (bool, optional): do you want print output? Defaults to False.
        Returns:
            (list[floats]): the training losses
        """   
        train_losses = []
        for r in range(rounds):
            client_trainloaders = self._make_client_trainloaders()
            round_loss = 0.0
            client_models = []
            for i in range(self.num_clients):
                client = BasicBaseline(device=self.device)
                client.set_trainloader(client_trainloaders[i])
                client.model.load_state_dict(self.model.state_dict())
                loss = client.train(
                    num_epochs=num_epochs,
                    lr=lr,
                    malicious=malicious if i<= ml-1 else 0,
                    verbose=verbose
                )[-1]
                client_models.append(client.model)
                round_loss += loss
                if verbose:
                    print(f"--> client {i} trained, round {r} \t final loss: {round(loss, 3)}\n")
            train_losses.append(round_loss / self.num_clients)
            self.aggregate(self.model, client_models)
        return train_losses

    def _make_client_trainloaders(self):
        """
        helper function to create client trainloader splits
        Returns:
            (list[torch.utils.data.Dataloader]): a list of dataloaders for the split data
        """        
        trainset_split = random_split(self.trainset, [int(len(self.trainset) / self.num_clients) for _ in range(self.num_clients)])
        return [DataLoader(x, batch_size=self.batch_size, shuffle=True) for x in trainset_split]

    @staticmethod
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
            global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()*upscale*(i <= ml -1) for i in range(len(client_models))], 0).mean(0)
        global_model.load_state_dict(global_dict)
        for model in client_models:
            model.load_state_dict(global_model.state_dict())
            

if __name__ == "__main__":
    # device = "cuda:0" #GPU
    device = "cuda:0" #CPU

    batch_size = 32
    lr = 0.001
    num_epochs = 2
    num_clients = 10
    rounds = 1
    verbose = True

    #Threat Model
    ml =1 #Number of malicious participants
    malicious = 2 # 0 no attack, 1 if universal attack, #2 if target attack
    tl = 4 #Target label in a targeted attack
    tc = 7  #Target missclassificatio.  Here, Classify sneakers as Dress
    upscale = 7 #Scale factor for parameters update

    # basic_baseline = BasicBaseline(device=device)
    # basic_baseline.load_data()
    # print(basic_baseline.train(
    #     num_epochs=num_epochs, 
    #     verbose=True))
    # print(basic_baseline.test())

    federated_baseline = FederatedBaseline(num_clients=num_clients,device="cuda:0")
    federated_baseline.load_data()
    print(federated_baseline.train(
        num_epochs=num_epochs, 
        rounds=rounds, 
        lr=lr, 
        verbose=verbose))
    print(federated_baseline.test())





