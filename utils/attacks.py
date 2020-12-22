

import torch
import numpy as np
<<<<<<< HEAD
from torch import optim, nn

from utils.models import AttackGenerator, FashionMNISTCNN
from utils.basics import load_model, uap_train
from utils.models import UAP


class Attack:
    def train(self, target_model, dataloader):
        return self
    
    def __repr__(self):
        return f"(attack=Attack)"


class NoAttack(Attack):
    def __init__(self):
        super(NoAttack, self).__init__()
        self.requires_training = False

    def run(self, inputs, labels):
        return inputs, labels
=======
from UAP_utils import UAP,trainUAP

class NoAttack:
    def run(self, labels):
        self.type = 0
        return labels
>>>>>>> a0c26214891a00d32d425b95229420879d489357

    def __repr__(self):
        return f"(attack=NoAttack)"


class RandomAttack(Attack):
    def __init__(self, num_classes):
        super(RandomAttack, self).__init__()
        self.num_classes = num_classes
        self.type = 0
    def run(self, labels):
        labels = torch.randint(0, self.num_classes, (np.size(labels, axis=0),))
        return labels
    
    def __repr__(self):
        return f"(attack=RandomAttack, num_classes={self.num_classes})"

class TargetedAttack:
    def __init__(self, target_label, target_class):
        self.target_label = target_label
        self.target_class = target_class
        self.type = 0
    def run(self, labels):
        labels[labels == self.target_label] = self.target_class
        return labels

    def __repr__(self):
        return f"(attack=TargetedAttack, target_label={self.target_label}, target_class={self.target_class})"
    
class UAPAttack:
    def __init__(self, target_label =3):
        self.target_label = target_label
        self.type = 1
    def run(self, data_loader, target_network,Cuda = True):       
        generator = UAP()
        target_network = target_network
        target_network.eval() 

import torch
import numpy as np
from utils.UAP_utils import UAP,trainUAP

class NoAttack:
    def __init__(self):
       self.type = 0
       
    def run(self, labels):
        return labels

    def __repr__(self):
        return f"(attack=NoAttack)"

<<<<<<< HEAD
    def run(self, inputs, labels):
=======
class RandomAttack:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.type = 0
    def run(self, labels):
>>>>>>> a0c26214891a00d32d425b95229420879d489357
        labels = torch.randint(0, self.num_classes, (np.size(labels, axis=0),))
        return inputs, labels
    
    def __repr__(self):
        return f"(attack=RandomAttack, num_classes={self.num_classes})"


class TargetedAttack(Attack):
    def __init__(self, target_label, target_class):
        super(TargetedAttack, self).__init__()
        self.target_label = target_label
        self.target_class = target_class
        self.type = 0
    def run(self, labels):
        labels[labels == self.target_label] = self.target_class
        return labels

<<<<<<< HEAD
    def run(self, inputs, labels):
=======
    def __repr__(self):
        return f"(attack=TargetedAttack, target_label={self.target_label}, target_class={self.target_class})"
    
class UAPAttack:
    def __init__(self, target_label =3):
        self.target_label = target_label
        self.type = 1
    def run(self, data_loader, target_network,Cuda = True):       
        generator = UAP()
        target_network = target_network
        target_network.eval() 
        if Cuda:
            generator.cuda()
        
        
        trainUAP(data_loader,
            generator,
            target_network)
        

        return generator
    
    def __repr__(self):
        return f"(attack=UAP,target_label={self.target_label})"


import torch
import numpy as np
from utils.UAP_utils import UAP,trainUAP

class NoAttack:
    def __init__(self):
       self.type = 0
       
    def run(self, labels):
        return labels

    def __repr__(self):
        return f"(attack=NoAttack)"

class RandomAttack:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.type = 0
    def run(self, labels):
        labels = torch.randint(0, self.num_classes, (np.size(labels, axis=0),))
        return labels
    
    def __repr__(self):
        return f"(attack=RandomAttack, num_classes={self.num_classes})"

class TargetedAttack:
    def __init__(self, target_label, target_class):
        self.target_label = target_label
        self.target_class = target_class
        self.type = 0
    def run(self, labels):
>>>>>>> a0c26214891a00d32d425b95229420879d489357
        labels[labels == self.target_label] = self.target_class
        return inputs, labels

    def __repr__(self):
        return f"(attack=TargetedAttack, target_label={self.target_label}, target_class={self.target_class})"
<<<<<<< HEAD


class UAPAttack(Attack):
    def __init__(self, target_label):
        super(UAPAttack, self).__init__()
        self.target_label = target_label        
        
    def train(self, target_model, dataloader, cuda=False):  
        """ overrides parent class train function """     
        self.generator = UAP()
        target_model.eval() 

        if cuda:
            self.generator.cuda()
        
        uap_train(dataloader,
            self.generator,
            target_model)  

        return self      

    def run(self, inputs, labels):
        for k in range(inputs.size(-4)):
            if labels[k] == self.target_label:
                inputs[k] = torch.squeeze(self.generator(inputs[k]).detach(), axis=0)
        
        return inputs, labels
    
    def __repr__(self):
        return f"(attack=UAPAttack, target_label={self.target_label})"


class GANAttack(Attack):
    def __init__(self, client_model):
        super(GANAttack, self).__init__()
        self.generator = AttackGenerator(input_dim=10, output_dim=1)
        self.discriminator = client_model

    def train(self, target_model, dataloader, num_epochs=5, z_dim=10, lr=1e-3, verbose=True):
        """ overrides parent class train function """  
        self.generator.train()
        self.discriminator.train()
        
        optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print_every = 10

        train_losses = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            epoch_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                labels = data[1]
                z = torch.rand((labels.size()[0], z_dim))

                optimizer.zero_grad()

                gen_fake = self.generator(z)
                dis_fake = self.discriminator(gen_fake)
                loss = criterion(dis_fake, labels)
                loss.backward()
                optimizer.step()

                if verbose:
                    running_loss += loss.item()
                    if i % print_every == 0 and i != 0:  
                        print(f"[epoch: {epoch}, datapoint: {i}] \t loss: {round(running_loss / print_every, 3)}")
                        running_loss = 0.0
                epoch_loss += loss.item()

            train_losses.append(epoch_loss / len(dataloader)) #this is buggy

        return train_losses


    def run(self, inputs, labels):
        return 

    def __repr__(self):
        return f"(attack=GANAttack)"

=======
    
class UAPAttack:
    def __init__(self, target_label =3):
        self.target_label = target_label
        self.type = 1
    def run(self, data_loader, target_network,Cuda = True):       
        generator = UAP()
        target_network = target_network
        target_network.eval() 
        if Cuda:
            generator.cuda()
        
        
        trainUAP(data_loader,
            generator,
            target_network)
        

        return generator
    
    def __repr__(self):
        return f"(attack=UAP,target_label={self.target_label})"

        if Cuda:
            generator.cuda()
        
        
        trainUAP(data_loader,
            generator,
            target_network)
        

        return generator
    
    def __repr__(self):
        return f"(attack=UAP,target_label={self.target_label})"
>>>>>>> a0c26214891a00d32d425b95229420879d489357
