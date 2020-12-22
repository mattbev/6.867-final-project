import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss


class NegativeCrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(NegativeCrossEntropy, self).__init__()

    def forward(self, x, target, uap, l=1):
        loss = F.cross_entropy(x, target, weight=None, ignore_index=-100)
        loss = loss - l * torch.norm(uap)
        return loss 