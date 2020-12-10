# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:30:16 2020
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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
from collections import OrderedDict

torch.manual_seed(1) #Set seed 
   



#Define the custom loss, I explain the technique in the slide, now it depends on the uap.
class NegativeCrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(NegativeCrossEntropy, self).__init__()

    def forward(self, input, target,uap,lam = 0.1):
        loss = F.cross_entropy(input.cuda(), target.cuda(), weight=None, ignore_index=-100)
        loss = loss - lam*torch.norm(uap)
        return loss 
    
    

#Define the UAP as an additive noise to images, if X is a image in tensor form, UAP(X) returns X+delta, where delta is the UAP.
class UAP(nn.Module):
    def __init__(self,
                shape=(28, 28),
                num_channels=1,# -*- coding: utf-8 -*-# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 23:30:16 2020
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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
from collections import OrderedDict

torch.manual_seed(1) #Set seed 
   



#Define the custom loss, I explain the technique in the slide, now it depends on the uap.
class NegativeCrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(NegativeCrossEntropy, self).__init__()

    def forward(self, input, target,uap,lam = 0.1):
        loss = F.cross_entropy(input.cuda(), target.cuda(), weight=None, ignore_index=-100)
        loss = loss - lam*torch.norm(uap)
        return loss 
    
    

#Define the UAP as an additive noise to images, if X is a image in tensor form, UAP(X) returns X+delta, where delta is the UAP.
class UAP(nn.Module):
    def __init__(self,
                shape=(28, 28),
                num_channels=1,
                mean=[0.5],
                std=[0.5],
                use_cuda=True):
        super(UAP, self).__init__()

        self.use_cuda = use_cuda
        self.num_channels = num_channels
        self.shape = shape
        self.uap = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))

        self.mean_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.mean_tensor[:,idx] *= mean[idx]
        if use_cuda:
            self.mean_tensor = self.mean_tensor.cuda()

        self.std_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.std_tensor[:,idx] *= std[idx]
        if use_cuda:
            self.std_tensor = self.std_tensor.cuda()

    def forward(self, x):
        uap = self.uap
        # Put image into original form
        orig_img = x * self.std_tensor + self.mean_tensor
        
        # Add uap to input
        adv_orig_img = orig_img + uap
        # Put image into normalized form
        adv_x = (adv_orig_img - self.mean_tensor)/self.std_tensor

        return adv_x



def trainUAP(data_loader,
            generator,
            target_network,
            epsilon = 1000,
            num_iterations = 1000,
            targeted = False,
            target_class = 4,
            print_freq=200,
            use_cuda=True):
    #Define the parameters
    criterion =NegativeCrossEntropy()
    optimizer = torch.optim.Adam(generator.parameters(), lr= 0.001)
    # switch to train mode
    generator.train()
    target_network.eval()
    
    
    data_iterator = iter(data_loader)
    
    iteration=0
    while (iteration<num_iterations):
        try:
            input, target = next(data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)
        
            #Class specific UAP
            ind  = []
            for k in range(input.size(-4)):       
                if target[k]  == target_class:
                    ind.append(k)
            input = input[ind]
            target = target[ind]
        
        
        
        
        
        if use_cuda:
            target = target.cuda()
            input = input.cuda()
        
        
        # compute output
            output = target_network(generator(input))
            loss = criterion(output, target,generator.uap)
        
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Projection
        generator.uap.data = torch.clamp(generator.uap.data, -epsilon, epsilon)
            
        iteration+=1
        if iteration % 100 == 99:    # print every 100 ioteration
            print('Optimization Iteration %d of %d' %(iteration,num_iterations))

    
         
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    
    
def metrics_evaluate(data_loader, generator,target_model, targeted=False, target_class=4, log=None, use_cuda=True):
    perturbed_model = nn.Sequential(OrderedDict([('generator', generator), ('target_model', target_model)]))
    perturbed_model = torch.nn.DataParallel(perturbed_model, device_ids=list(range(1)))
    # switch to evaluate mode
    perturbed_model.eval()
    perturbed_model.module.generator.eval()
    perturbed_model.module.target_model.eval()

    clean_acc = AverageMeter()
    perturbed_acc = AverageMeter()
    attack_success_rate = AverageMeter() # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
    if targeted:
        all_to_target_success_rate = AverageMeter() # The ratio of samples going to the sink classes
        all_to_target_success_rate_filtered = AverageMeter()

    total_num_samples = 0
    num_same_classified = 0
    num_diff_classified = 0

    for input, gt in data_loader:
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            clean_output = target_model(input)
            pert_output = perturbed_model(input)

        correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
        cl_acc = accuracy(clean_output.data, gt, topk=(1,))
        clean_acc.update(cl_acc[0].item(), input.size(0))
        pert_acc = accuracy(pert_output.data, gt, topk=(1,))
        perturbed_acc.update(pert_acc[0].item(), input.size(0))

        # Calculating Fooling Ratio params
        clean_out_class = torch.argmax(clean_output, dim=-1)
        pert_out_class = torch.argmax(pert_output, dim=-1)

        total_num_samples += len(clean_out_class)
        num_same_classified += torch.sum(clean_out_class == pert_out_class).cpu().numpy()
        num_diff_classified += torch.sum(~(clean_out_class == pert_out_class)).cpu().numpy()

        if torch.sum(correctly_classified_mask)>0:
            with torch.no_grad():
                pert_output_corr_cl = perturbed_model(input[correctly_classified_mask])
            attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
            attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))


        # Calculate Absolute Accuracy Drop
        aad_source = clean_acc.avg - perturbed_acc.avg
        # Calculate Relative Accuracy Drop
        if clean_acc.avg !=0:
            rad_source = (clean_acc.avg - perturbed_acc.avg)/clean_acc.avg * 100.
        else:
            rad_source = 0.
        # Calculate fooling ratio
        fooling_ratio = num_diff_classified/total_num_samples * 100.

        if targeted:
            # 2. How many of all samples go the sink class (Only relevant for others loader)
            target_cl = torch.ones_like(gt) * target_class
            all_to_target_succ_rate = accuracy(pert_output, target_cl, topk=(1,))
            all_to_target_success_rate.update(all_to_target_succ_rate[0].item(), pert_output.size(0))

            # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
            # Filter all idxs which are not belonging to sink class
            non_target_class_idxs = [i != target_class for i in gt]
            non_target_class_mask = torch.Tensor(non_target_class_idxs)==True
            if torch.sum(non_target_class_mask)>0:
                gt_non_target_class = gt[non_target_class_mask]
                pert_output_non_target_class = pert_output[non_target_class_mask]

                target_cl = torch.ones_like(gt_non_target_class) * target_class
                all_to_target_succ_rate_filtered = accuracy(pert_output_non_target_class, target_cl, topk=(1,))
                all_to_target_success_rate_filtered.update(all_to_target_succ_rate_filtered[0].item(), pert_output_non_target_class.size(0))
    print('\tClean model accuracy: %f %% ' %(clean_acc.avg))
    print('\tPerturbed model accuracy:  %f  %%' %(perturbed_acc.avg))
    print('\tAbsolute Accuracy Drop:  %f  %%' %(aad_source))
    print('\tRelative Accuracy Drop: %f  %%' %(rad_source))
    print('\tAttack Success Rate: %f  %%' %(100-attack_success_rate.avg))
    print('\tFooling Ratio: %f  %%' %(fooling_ratio))
  
    def plotfigs(generator,trainloader):
        for data, _ in trainloader:
            # Rearrange batch to be the shape of [B, C, W * H]
            inputs = data[0].cuda()
            inputs2 = generator(inputs)
            # Update total number of images
            t=  inputs2.detach()
            t = torch.squeeze(t, 0)
            imshow(t.cpu())
            imshow(inputs.cpu())
    
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

"""
Created on Wed Dec  9 23:30:16 2020
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

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss
import numpy as np
from collections import OrderedDict

torch.manual_seed(1) #Set seed 
   



#Define the custom loss, I explain the technique in the slide, now it depends on the uap.
class NegativeCrossEntropy(_WeightedLoss):
    def __init__(self,):
        super(NegativeCrossEntropy, self).__init__()

    def forward(self, input, target,uap,lam = 0.1):
        loss = F.cross_entropy(input.cuda(), target.cuda(), weight=None, ignore_index=-100)
        loss = loss - lam*torch.norm(uap)
        return loss 
    
    

#Define the UAP as an additive noise to images, if X is a image in tensor form, UAP(X) returns X+delta, where delta is the UAP.
class UAP(nn.Module):
    def __init__(self,
                shape=(28, 28),
                num_channels=1,
                mean=[0.5],
                std=[0.5],
                use_cuda=True):
        super(UAP, self).__init__()

        self.use_cuda = use_cuda
        self.num_channels = num_channels
        self.shape = shape
        self.uap = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))

        self.mean_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.mean_tensor[:,idx] *= mean[idx]
        if use_cuda:
            self.mean_tensor = self.mean_tensor.cuda()

        self.std_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.std_tensor[:,idx] *= std[idx]
        if use_cuda:
            self.std_tensor = self.std_tensor.cuda()

    def forward(self, x):
        uap = self.uap
        # Put image into original form
        orig_img = x * self.std_tensor + self.mean_tensor
        
        # Add uap to input
        adv_orig_img = orig_img + uap
        # Put image into normalized form
        adv_x = (adv_orig_img - self.mean_tensor)/self.std_tensor

        return adv_x



def trainUAP(data_loader,
            generator,
            target_network,
            epsilon = 1000,
            num_iterations = 1000,
            targeted = False,
            target_class = 4,
            print_freq=200,
            use_cuda=True):
    #Define the parameters
    criterion =NegativeCrossEntropy()
    optimizer = torch.optim.Adam(generator.parameters(), lr= 0.001)
    # switch to train mode
    generator.train()
    target_network.eval()
    
    
    data_iterator = iter(data_loader)
    
    iteration=0
    while (iteration<num_iterations):
        try:
            input, target = next(data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)
        
            #Class specific UAP
            ind  = []
            for k in range(input.size(-4)):       
                if target[k]  == target_class:
                    ind.append(k)
            input = input[ind]
            target = target[ind]
        
        
        
        
        
        if use_cuda:
            target = target.cuda()
            input = input.cuda()
        
        
        # compute output
            output = target_network(generator(input))
            loss = criterion(output, target,generator.uap)
        
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Projection
        generator.uap.data = torch.clamp(generator.uap.data, -epsilon, epsilon)
            
        iteration+=1
        if iteration % 100 == 99:    # print every 100 ioteration
            print('Optimization Iteration %d of %d' %(iteration,num_iterations))

    
         
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    
    
def metrics_evaluate(data_loader, generator,target_model, targeted=False, target_class=4, log=None, use_cuda=True):
    perturbed_model = nn.Sequential(OrderedDict([('generator', generator), ('target_model', target_model)]))
    perturbed_model = torch.nn.DataParallel(perturbed_model, device_ids=list(range(1)))
    # switch to evaluate mode
    perturbed_model.eval()
    perturbed_model.module.generator.eval()
    perturbed_model.module.target_model.eval()

    clean_acc = AverageMeter()
    perturbed_acc = AverageMeter()
    attack_success_rate = AverageMeter() # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
    if targeted:
        all_to_target_success_rate = AverageMeter() # The ratio of samples going to the sink classes
        all_to_target_success_rate_filtered = AverageMeter()

    total_num_samples = 0
    num_same_classified = 0
    num_diff_classified = 0

    for input, gt in data_loader:
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            clean_output = target_model(input)
            pert_output = perturbed_model(input)

        correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
        cl_acc = accuracy(clean_output.data, gt, topk=(1,))
        clean_acc.update(cl_acc[0].item(), input.size(0))
        pert_acc = accuracy(pert_output.data, gt, topk=(1,))
        perturbed_acc.update(pert_acc[0].item(), input.size(0))

        # Calculating Fooling Ratio params
        clean_out_class = torch.argmax(clean_output, dim=-1)
        pert_out_class = torch.argmax(pert_output, dim=-1)

        total_num_samples += len(clean_out_class)
        num_same_classified += torch.sum(clean_out_class == pert_out_class).cpu().numpy()
        num_diff_classified += torch.sum(~(clean_out_class == pert_out_class)).cpu().numpy()

        if torch.sum(correctly_classified_mask)>0:
            with torch.no_grad():
                pert_output_corr_cl = perturbed_model(input[correctly_classified_mask])
            attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
            attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))


        # Calculate Absolute Accuracy Drop
        aad_source = clean_acc.avg - perturbed_acc.avg
        # Calculate Relative Accuracy Drop
        if clean_acc.avg !=0:
            rad_source = (clean_acc.avg - perturbed_acc.avg)/clean_acc.avg * 100.
        else:
            rad_source = 0.
        # Calculate fooling ratio
        fooling_ratio = num_diff_classified/total_num_samples * 100.

        if targeted:
            # 2. How many of all samples go the sink class (Only relevant for others loader)
            target_cl = torch.ones_like(gt) * target_class
            all_to_target_succ_rate = accuracy(pert_output, target_cl, topk=(1,))
            all_to_target_success_rate.update(all_to_target_succ_rate[0].item(), pert_output.size(0))

            # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
            # Filter all idxs which are not belonging to sink class
            non_target_class_idxs = [i != target_class for i in gt]
            non_target_class_mask = torch.Tensor(non_target_class_idxs)==True
            if torch.sum(non_target_class_mask)>0:
                gt_non_target_class = gt[non_target_class_mask]
                pert_output_non_target_class = pert_output[non_target_class_mask]

                target_cl = torch.ones_like(gt_non_target_class) * target_class
                all_to_target_succ_rate_filtered = accuracy(pert_output_non_target_class, target_cl, topk=(1,))
                all_to_target_success_rate_filtered.update(all_to_target_succ_rate_filtered[0].item(), pert_output_non_target_class.size(0))
    print('\tClean model accuracy: %f %% ' %(clean_acc.avg))
    print('\tPerturbed model accuracy:  %f  %%' %(perturbed_acc.avg))
    print('\tAbsolute Accuracy Drop:  %f  %%' %(aad_source))
    print('\tRelative Accuracy Drop: %f  %%' %(rad_source))
    print('\tAttack Success Rate: %f  %%' %(100-attack_success_rate.avg))
    print('\tFooling Ratio: %f  %%' %(fooling_ratio))
  
    def plotfigs(generator,trainloader):
        for data, _ in trainloader:
            # Rearrange batch to be the shape of [B, C, W * H]
            inputs = data[0].cuda()
            inputs2 = generator(inputs)
            # Update total number of images
            t=  inputs2.detach()
            t = torch.squeeze(t, 0)
            imshow(t.cpu())
            imshow(inputs.cpu())
    
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

                mean=[0.5],
                std=[0.5],
                use_cuda=True):
        super(UAP, self).__init__()

        self.use_cuda = use_cuda
        self.num_channels = num_channels
        self.shape = shape
        self.uap = nn.Parameter(torch.zeros(size=(num_channels, *shape), requires_grad=True))

        self.mean_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.mean_tensor[:,idx] *= mean[idx]
        if use_cuda:
            self.mean_tensor = self.mean_tensor.cuda()

        self.std_tensor = torch.ones(1, num_channels, *shape)
        for idx in range(num_channels):
            self.std_tensor[:,idx] *= std[idx]
        if use_cuda:
            self.std_tensor = self.std_tensor.cuda()

    def forward(self, x):
        uap = self.uap
        # Put image into original form
        orig_img = x * self.std_tensor + self.mean_tensor
        
        # Add uap to input
        adv_orig_img = orig_img + uap
        # Put image into normalized form
        adv_x = (adv_orig_img - self.mean_tensor)/self.std_tensor

        return adv_x



def trainUAP(data_loader,
            generator,
            target_network,
            epsilon = 1000,
            num_iterations = 1000,
            targeted = False,
            target_class = 4,
            print_freq=200,
            use_cuda=True):
    #Define the parameters
    criterion =NegativeCrossEntropy()
    optimizer = torch.optim.Adam(generator.parameters(), lr= 0.001)
    # switch to train mode
    generator.train()
    target_network.eval()
    
    
    data_iterator = iter(data_loader)
    
    iteration=0
    while (iteration<num_iterations):
        try:
            input, target = next(data_iterator)
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iterator = iter(data_loader)
            input, target = next(data_iterator)
        
            #Class specific UAP
            ind  = []
            for k in range(input.size(-4)):       
                if target[k]  == target_class:
                    ind.append(k)
            input = input[ind]
            target = target[ind]
        
        
        
        
        
        if use_cuda:
            target = target.cuda()
            input = input.cuda()
        
        
        # compute output
            output = target_network(generator(input))
            loss = criterion(output, target,generator.uap)
        
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Projection
        generator.uap.data = torch.clamp(generator.uap.data, -epsilon, epsilon)
            
        iteration+=1
        if iteration % 100 == 99:    # print every 100 ioteration
            print('Optimization Iteration %d of %d' %(iteration,num_iterations))

    
         
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    
    
def metrics_evaluate(data_loader, generator,target_model, targeted=False, target_class=4, log=None, use_cuda=True):
    perturbed_model = nn.Sequential(OrderedDict([('generator', generator), ('target_model', target_model)]))
    perturbed_model = torch.nn.DataParallel(perturbed_model, device_ids=list(range(1)))
    # switch to evaluate mode
    perturbed_model.eval()
    perturbed_model.module.generator.eval()
    perturbed_model.module.target_model.eval()

    clean_acc = AverageMeter()
    perturbed_acc = AverageMeter()
    attack_success_rate = AverageMeter() # Among the correctly classified samples, the ratio of being different from clean prediction (same as gt)
    if targeted:
        all_to_target_success_rate = AverageMeter() # The ratio of samples going to the sink classes
        all_to_target_success_rate_filtered = AverageMeter()

    total_num_samples = 0
    num_same_classified = 0
    num_diff_classified = 0

    for input, gt in data_loader:
        if use_cuda:
            gt = gt.cuda()
            input = input.cuda()

        # compute output
        with torch.no_grad():
            clean_output = target_model(input)
            pert_output = perturbed_model(input)

        correctly_classified_mask = torch.argmax(clean_output, dim=-1).cpu() == gt.cpu()
        cl_acc = accuracy(clean_output.data, gt, topk=(1,))
        clean_acc.update(cl_acc[0].item(), input.size(0))
        pert_acc = accuracy(pert_output.data, gt, topk=(1,))
        perturbed_acc.update(pert_acc[0].item(), input.size(0))

        # Calculating Fooling Ratio params
        clean_out_class = torch.argmax(clean_output, dim=-1)
        pert_out_class = torch.argmax(pert_output, dim=-1)

        total_num_samples += len(clean_out_class)
        num_same_classified += torch.sum(clean_out_class == pert_out_class).cpu().numpy()
        num_diff_classified += torch.sum(~(clean_out_class == pert_out_class)).cpu().numpy()

        if torch.sum(correctly_classified_mask)>0:
            with torch.no_grad():
                pert_output_corr_cl = perturbed_model(input[correctly_classified_mask])
            attack_succ_rate = accuracy(pert_output_corr_cl, gt[correctly_classified_mask], topk=(1,))
            attack_success_rate.update(attack_succ_rate[0].item(), pert_output_corr_cl.size(0))


        # Calculate Absolute Accuracy Drop
        aad_source = clean_acc.avg - perturbed_acc.avg
        # Calculate Relative Accuracy Drop
        if clean_acc.avg !=0:
            rad_source = (clean_acc.avg - perturbed_acc.avg)/clean_acc.avg * 100.
        else:
            rad_source = 0.
        # Calculate fooling ratio
        fooling_ratio = num_diff_classified/total_num_samples * 100.

        if targeted:
            # 2. How many of all samples go the sink class (Only relevant for others loader)
            target_cl = torch.ones_like(gt) * target_class
            all_to_target_succ_rate = accuracy(pert_output, target_cl, topk=(1,))
            all_to_target_success_rate.update(all_to_target_succ_rate[0].item(), pert_output.size(0))

            # 3. How many of all samples go the sink class, except gt sink class (Only relevant for others loader)
            # Filter all idxs which are not belonging to sink class
            non_target_class_idxs = [i != target_class for i in gt]
            non_target_class_mask = torch.Tensor(non_target_class_idxs)==True
            if torch.sum(non_target_class_mask)>0:
                gt_non_target_class = gt[non_target_class_mask]
                pert_output_non_target_class = pert_output[non_target_class_mask]

                target_cl = torch.ones_like(gt_non_target_class) * target_class
                all_to_target_succ_rate_filtered = accuracy(pert_output_non_target_class, target_cl, topk=(1,))
                all_to_target_success_rate_filtered.update(all_to_target_succ_rate_filtered[0].item(), pert_output_non_target_class.size(0))
    print('\tClean model accuracy: %f %% ' %(clean_acc.avg))
    print('\tPerturbed model accuracy:  %f  %%' %(perturbed_acc.avg))
    print('\tAbsolute Accuracy Drop:  %f  %%' %(aad_source))
    print('\tRelative Accuracy Drop: %f  %%' %(rad_source))
    print('\tAttack Success Rate: %f  %%' %(100-attack_success_rate.avg))
    print('\tFooling Ratio: %f  %%' %(fooling_ratio))
  
    def plotfigs(generator,trainloader):
        for data, _ in trainloader:
            # Rearrange batch to be the shape of [B, C, W * H]
            inputs = data[0].cuda()
            inputs2 = generator(inputs)
            # Update total number of images
            t=  inputs2.detach()
            t = torch.squeeze(t, 0)
            imshow(t.cpu())
            imshow(inputs.cpu())
    
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
