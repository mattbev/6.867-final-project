import os
import pickle
import torch
import imageio
import itertools
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.metrics import confusion_matrix
from collections import OrderedDict


from utils.losses import NegativeCrossEntropy


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return imageio.imwrite(path, image)


def imshow(img):
    """
    show an image
    Args:
        img (TODO): the image to display
    """    
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.imshow(npimg, cmap='gray')
    plt.show()
    plt.savefig("figures/gan_output.png")
    

def generic_train(model, num_epochs, trainloader, optimizer, criterion, attack, device="cpu", verbose=False):
    """
    train a model
    Args:
        model (torch.nn.Module): the model to train
        num_epochs (int): the number of epochs
        trainloader (torch.utils.data.Dataloader): the training dataset dataloader
        optimizer (torch.optim.*): the function to optimize with
        criterion (torch.nn.*): the loss function
        attack (Attack): the attack model
        device (str or pytorch device, optional): where to evaluate pytorch variables. Defaults to "cpu".
        verbose (bool, optional): extended print statement? Defaults to False.
    Returns:
        (list[float]): the training loss per epoch 
    """ 
    print_every = 50
    if type(device) == str:  
        device = torch.device(device) 
    model = model.to(device)

    # train attack
    attack.train(model, trainloader)

    model.train()
    train_losses = []
    for epoch in range(num_epochs):  # loop over the dataset multiple time
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)# get the inputs; data is a list of [inputs, labels]
            inputs, labels = attack.run(inputs, labels)
            inputs.to(device) 
            labels.to(device)
            optimizer.zero_grad() # zero the parameter gradients

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if verbose:
                running_loss += loss.item()
                if i % print_every == 0 and i != 0:  
                    print(f"[epoch: {epoch}, datapoint: {i}] \t loss: {round(running_loss / print_every, 3)}")
                    running_loss = 0.0
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(trainloader)) #this is buggy

    return train_losses
            

def test_total_accurcy(model, testloader, device="cpu"):
    """
    compute the (pure) accuracy over a test set 
    Args:
        model (torch.nn.Module): [description]
        testloader (torch.utils.data.Dataloader): the test set dataloader
        device (str or pytorch device, optional): where to evaluate pytorch variables. Defaults to "cpu".
    Returns:
        (float): the accuracy
    """  
    if type(device) == str:  
        device = torch.device(device) 
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def test_class_accuracy(model, testloader, device="cpu"):
    """
    compute (pure) accuracy per class in the test set
    Args:
        model (torch.nn.Module): the model to evaluate
        testloader (torch.utils.data.Dataloader): the test set dataloader
        device (str or pytorch device, optional): where to evaluate pytorch variables. Defaults to "cpu".
    """    
    if type(device) == str:  
        device = torch.device(device) 
    class_correct = np.array([0. for i in range(10)])
    class_total = np.array([0. for i in range(10)])
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    return class_correct / class_total


def test_confusion_matrix(model, testloader, device="cpu"):
    predictions = torch.Tensor([])
    groundtruths = torch.Tensor([])
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions = torch.cat((predictions, predicted), dim=0)
            groundtruths = torch.cat((groundtruths, labels), dim=0)
    return confusion_matrix(groundtruths, predictions)
    

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return plt


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def save_model(model, name):
    save_dir = os.path.join("saved_models", name)
    torch.save(model.state_dict(), os.path.join("saved_models", f"{name}.pkl"))


def load_model(model, name):
    model.load_state_dict(torch.load(os.path.join("saved_models", f"{name}.pkl")))
    return model


def uap_train(data_loader, generator, target_network, epsilon=1000, num_iterations=1000, targeted=False,
            target_class=4, print_freq=200, use_cuda=False):
            
    #Define the parameters
    criterion = NegativeCrossEntropy()
    optimizer = torch.optim.Adam(generator.parameters(), lr= 0.001)
    
    # switch to train mode
    generator.train()
    target_network.eval()
    data_iterator = iter(data_loader)
    iteration=0
    while iteration < num_iterations:
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
        if iteration % 100 == 0:    # print every 100 ioteration
            print('Optimization Iteration %d of %d' %(iteration,num_iterations))



def precision_k(output, target, topk=(1,)):
    """Computes the precision for the specified values of k"""
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