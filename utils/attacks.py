

import torch
import numpy as np
from UAP_utils import UAP,trainUAP

class NoAttack:
    def run(self, labels):
        self.type = 0
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
