# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import torch
import numpy as np
import pandas as pd
import torchvision.datasets as datasets
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split

from utils.models import FashionMNISTCNN
from utils.basics import generic_train, test_total_accuracy, test_class_accuracy, save_model
from utils.attacks import NoAttack, RandomAttack, TargetedAttack, UAPAttack
from utils.defenses import NoDefense, FlippedLabelsDefense

torch.manual_seed(1) #Set seed 

class Baseline:
    def __init__(self, device="cpu"):
        """
        Baseline parent class
        Args:
            device (str, optional): where to run pytorch on. Defaults to "cpu".
        """        
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
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, ), (0.5, ),)]) # normalize to [-1,1]
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
        """
        Basic CNN Baseline
        Args:
            device (str, optional): the device to run pytorch on. Defaults to "cpu".
        """        
        super(BasicBaseline, self).__init__(device=device)
        

    def set_trainloader(self, trainloader):
        """
        set the trainloader for the model
        Args:
            trainloader (torch.utils.data.Dataloader): the training data data loader
        """        
        self.trainloader = trainloader


    def configure_attack(self, attack=NoAttack()):
        """
        configure malicious attacks on the model
        Args:
            attack (Attack, optional): The attack to apply to the model. Defaults to NoAttack().
        """        
        self.attack = attack


    def train(self, num_epochs, lr=1e-3, verbose=False, print_summary=True):
        """
        train the basic baseline model
        Args:
            num_epochs (int): the number of epochs.
            lr (float, optional): the learning rate. Defaults to 1e-3.
            verbose (bool, optional): do you want print output? Defaults to False.
            print_summary (bool, optional): print the hyperparameters. Defaults to True.
        Returns:
            (list[floats]): the training losses
        """  

        if print_summary:
            print(f"Training BasicBaseline model.")	            
            print("========== HYPERPARAMETERS ==========")
            print(f"num_epochs: {num_epochs}")	            
            print(f"lr: {lr}")	
            print(f"attack: {self.attack}")
            print("\n")

        criterion, optimizer = self._make_optimizer_and_loss(lr)
        return generic_train(
            model=self.model, 
            num_epochs=num_epochs, 
            trainloader=self.trainloader, 
            optimizer=optimizer, 
            criterion=criterion,  
            attack=self.attack,
            device=self.device, 
            verbose=verbose)



class FederatedBaseline(Baseline):
    def __init__(self, num_clients, device="cpu"):
        """
        Federated CNN baseline model
        Args:
            num_clients (int): number of clients for federated learning
            device (str, optional): where to run pytorch on. Defaults to "cpu".
        """        
        super(FederatedBaseline, self).__init__(device=device)
        self.num_clients = num_clients
        self.round_log = []


    def configure_attack(self, attack=NoAttack(), num_malicious=0):
        """
        configure malicious attacks against the model from clients
        Args:
            attack (Attack, optional): the attack type. Defaults to NoAttack().
            num_malicious (int, optional): number of malicious clients using this attack. Defaults to 0.
        """        
        assert num_malicious <= self.num_clients, "num_malicious must be <= num_clients"
        self.attack = attack
        self.num_malicious = num_malicious
        self.attacks = [attack for i in range(num_malicious)]
        self.attacks.extend([NoAttack() for i in range(self.num_clients - num_malicious)])

    
    def manual_attack(self, attack_list):
        """
        manually set the attacks
        Args:
            attack_list (iterable[Attack]): the attacks
        """        
        assert len(attack_list) == self.num_clients, "len(attack_list) must be == num_clients"
        self.attacks = attack_list


    def configure_defense(self, defense):
        """
        configure the federated learning defense
        Args:
            defense (Defense): the defense
        """        
        self.defense = defense


    def train(self, num_epochs, rounds=1, lr=1e-3, malicious_upscale=1.0, log=True, verbose=False, print_summary=True):
        """
        train the federated baseline model
        Args:
            num_epochs (int): the number of epochs
            rounds (int, optional): the number of rounds to train clients. Defaults to 1.
            lr (float, optional): the learning rate. Defaults to 1e-3.
            malicious_upscale (float, optional): scale factor for parameter updates of the malicious models.
            log (boolean, optional): to log the round-wise accuracies. Defaults to True.
            verbose (bool, optional): do you want print output? Defaults to False.
            print_summary (bool, optional): print the hyperparameters. Defaults to True.
        Returns:
            (list[floats]): the training losses
        """   

        if print_summary:
            print(f"Training FederatedBaseline model with {self.num_clients} clients.")	            
            print("========== HYPERPARAMETERS ==========")
            print(f"num_clients: {self.num_clients}")
            print(f"num_epochs: {num_epochs}")	            
            print(f"rounds: {rounds}")	            
            print(f"lr: {lr}")	
            print(f"num_malicious: {self.num_malicious}")
            print(f"attack: {self.attack}")
            print(f"malicious_upscale: {malicious_upscale}")
            print(f"defense: {self.defense}")
            print(f"log: {log}")
            print("\n")         

        train_losses = []
        for r in range(rounds):
            client_trainloaders = self._make_client_trainloaders()
            round_loss = 0.0
            client_models = []
            for i in range(self.num_clients):
                client = BasicBaseline(device=self.device)
                client.set_trainloader(client_trainloaders[i])
                client.model.load_state_dict(self.model.state_dict())
                client.configure_attack(attack=self.attacks[i])
                loss = client.train(
                    num_epochs=num_epochs,
                    lr=lr,
                    verbose=verbose,
                    print_summary=False
                )[-1]
                client_models.append(client.model)
                round_loss += loss
                if verbose:
                    print(f"--> client {i} trained, round {r} \t final loss: {round(loss, 3)}\n")
            train_losses.append(round_loss / self.num_clients)
            self._aggregate(client_models, malicious_upscale)
            if log:
                accuracies = self.test()
                print(accuracies)
                overall, classwise = accuracies
                total = classwise.tolist()
                total.insert(0, overall)
                self.round_log.append(total)
        return train_losses


    def _make_client_trainloaders(self):
        """
        helper function to create client trainloader splits
        Returns:
            (list[torch.utils.data.Dataloader]): a list of dataloaders for the split data
        """        
        trainset_split = random_split(self.trainset, [int(len(self.trainset) / self.num_clients) for _ in range(self.num_clients)])
        return [DataLoader(x, batch_size=self.batch_size, shuffle=True) for x in trainset_split]


    def _aggregate(self, client_models, malicious_upscale):
        """
        global parameter updates aggregation.
        Args:
            client_models (list[torch.nn.Module]): the client models
            malicious_upscale (float): scale factor for parameter updates
        """    
        ### take simple mean of the weights of models ###
        safe_clients = self.defense.run(self.model, client_models, plot_name="fig.png")
        global_dict = self.model.state_dict()
        for k in global_dict.keys():
            update = [safe_clients[i].state_dict()[k].float() for i in range(len(safe_clients))]
            update[:self.num_malicious] *= malicious_upscale
            global_dict[k] = torch.stack(update, axis=0).mean(axis=0)
        self.model.load_state_dict(global_dict)
            




if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 32
    lr = 1e-3
    num_epochs = 2
    num_clients = 10
    rounds = 1
    verbose = True

    #Threat Model
    malicious_upscale = 20 #Scale factor for parameters update
    num_malicious = 2
    # attack = NoAttack()
    # attack = RandomAttack(num_classes=10)
    # attack = TargetedAttack(target_label=3, target_class=7)
    attack = UAPAttack(target_label=3)

    #Defense Model
    # defense = NoDefense()
    defense = FlippedLabelsDefense(num_classes=1)
    
    args = sys.argv
    assert len(args) == 2, "incorrect number of arguments."
    test = args[1]

    if test == "basic":
        basic_baseline = BasicBaseline(device=device)
        basic_baseline.load_data()
        basic_baseline.configure_attack(attack=attack)

        print(basic_baseline.train(
            num_epochs=num_epochs, 
            lr=lr,
            verbose=True))
        
        print(basic_baseline.test())

        # save_model(basic_baseline.model, "basic_25epochs_NoAttack")

    elif test == "federated":
        federated_baseline = FederatedBaseline(num_clients=num_clients, device=device)
        federated_baseline.load_data()
        federated_baseline.configure_attack(attack=attack, num_malicious=num_malicious)
        federated_baseline.configure_defense(defense=defense)
        
        print(federated_baseline.train(
            num_epochs=num_epochs, 
            rounds=rounds, 
            lr=lr, 
            malicious_upscale=malicious_upscale,
            verbose=verbose))
        
        print(federated_baseline.test())

        # save_model(federated_baseline.model, "defense_cm")
        # logs = federated_baseline.round_log
        # columns = ["overall"] + [f"class{i}" for i in range(10)]
        # df = pd.DataFrame(np.array(logs), columns=columns)
        # df.to_csv("random_attack_with_defense_10rounds.csv")
    else:
        print("incorrect arguments.")





