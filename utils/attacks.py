import torch
import numpy as np
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

