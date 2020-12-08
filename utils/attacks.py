import torch
import numpy as np

class NoAttack:
    def run(self, labels):
        return labels

    def __repr__(self):
        return f"(attack=NoAttack)"

class RandomAttack:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def run(self, labels):
        labels = torch.randint(0, self.num_classes, (np.size(labels, axis=0),))
        return labels
    
    def __repr__(self):
        return f"(attack=RandomAttack, num_classes={self.num_classes})"

class TargetedAttack:
    def __init__(self, target_label, target_class):
        self.target_label = target_label
        self.target_class = target_class

    def run(self, labels):
        labels[labels == self.target_label] = self.target_class
        return labels

    def __repr__(self):
        return f"(attack=TargetedAttack, target_label={self.target_label}, target_class={self.target_class})"
