import torch
import numpy as np

from utils.models import AttackGenerator, FashionMNISTCNN
from utils.basics import load_model
from utils.UAP_utils import UAP, trainUAP


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

    def __repr__(self):
        return f"(attack=NoAttack)"


class RandomAttack(Attack):
    def __init__(self, num_classes):
        super(RandomAttack, self).__init__()
        self.num_classes = num_classes

    def run(self, inputs, labels):
        labels = torch.randint(0, self.num_classes, (np.size(labels, axis=0),))
        return inputs, labels
    
    def __repr__(self):
        return f"(attack=RandomAttack, num_classes={self.num_classes})"


class TargetedAttack(Attack):
    def __init__(self, target_label, target_class):
        super(TargetedAttack, self).__init__()
        self.target_label = target_label
        self.target_class = target_class

    def run(self, inputs, labels):
        labels[labels == self.target_label] = self.target_class
        return inputs, labels

    def __repr__(self):
        return f"(attack=TargetedAttack, target_label={self.target_label}, target_class={self.target_class})"


class UAPAttack(Attack):
    def __init__(self, target_label):
        super(UAPAttack, self).__init__()
        self.target_label = target_label        
        
    def train(self, target_model, dataloader, cuda=True):  
        """ overrides parent class train function """     
        self.generator = UAP()
        target_model.eval() 

        if cuda:
            self.generator.cuda()
        
        trainUAP(dataloader,
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
        self.generator = AttackGenerator(input_dim=10, output_dim=7*7*32)
        self.discriminator = client_model

    def train(self, target_model, dataloader):
        """ overrides parent class train function """  
        return self

    def run(self, inputs, labels):
        return 

