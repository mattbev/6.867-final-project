from collections import defaultdict

import torch
import torch.utils.data

from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.resnet import ResNet18
from models.dconv import _netG, _netD
from models.word_model import RNNModel
from utils.text_load import *
from utils.utils import SubsetSampler
from data.svhn import SvhnDataset

logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0


class ImageHelper(Helper):

    def poison(self):
        return

    def create_model(self):
        nc = 1
        # local_model = ResNet18(name='Local', created_time=self.params['current_time'])
        local_model = _netD(64, .2, nc, .5, 10, name='Local', created_time=self.params['current_time'])
        local_model.apply(self._weights_init)
        local_model.cuda()
        # target_model = ResNet18(name='Target', created_time=self.params['current_time'])
        target_model = _netD(64, .2, nc, .5, 10, name='Target', created_time=self.params['current_time'])
        target_model.apply(self._weights_init)
        target_model.cuda()
        if self.params['resumed_model']:
            loaded_params = torch.load(f"saved_models/{self.params['resumed_model']}")
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

        g_model = _netG(100, 64, .2, nc, True)
        g_model.apply(self._weights_init)
        g_model.cuda()
        self.g_model = g_model

    def sample_dirichlet_train_data(self, no_participants, alpha=0.9):
        """
            Input: Number of participants and alpha (param for distribution)
            Output: A list of indices denoting data in CIFAR training set.
            Requires: cifar_classes, a preprocessed class-indice dictionary.
            Sample Method: take a uniformly sampled 10-dimension vector as parameters for
            dirichlet distribution to sample number of images in each class.
        """

        cifar_classes = {}
        for ind, x in enumerate(self.train_dataset):
            _, label = x
            # for mnist label need to convert to in
            label = label.item()
            if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
                continue
            if label in cifar_classes:
                cifar_classes[label].append(ind)
            else:
                cifar_classes[label] = [ind]

        class_size = len(cifar_classes[0])
        per_participant_list = defaultdict(list)
        no_classes = len(cifar_classes.keys())

        for n in range(no_classes):
            random.shuffle(cifar_classes[n])
            sampled_probabilities = class_size * np.random.dirichlet(
                np.array(no_participants * [alpha]))
            for user in range(no_participants):
                no_imgs = int(round(sampled_probabilities[user]))
                sampled_list = cifar_classes[n][:min(len(cifar_classes[n]), no_imgs)]
                per_participant_list[user].extend(sampled_list)
                cifar_classes[n] = cifar_classes[n][min(len(cifar_classes[n]), no_imgs):]

        return per_participant_list

    def load_data(self):
        logger.info('Loading data')

        ### data load
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # self.train_dataset = datasets.CIFAR10('./data/cifar/', train=True, download=True,transform=transform_train)
        # self.test_dataset = datasets.CIFAR10('./data/cifar/', train=False, transform=transform_test)

        self.train_dataset = SvhnDataset(32, 'train', './data/cifar/', True)
        self.test_dataset = SvhnDataset(32, 'test', './data/cifar/', True)
        print(len(self.train_dataset), len(self.test_dataset))

        self.gen_dataset = SvhnDataset(32, 'train', './data/cifar/', True)
        self.gen_dataset.svhn_dataset.train_labels = np.array(self.gen_dataset.svhn_dataset.train_labels)
        self.gen_dataset.svhn_dataset.train_data = self.gen_dataset.svhn_dataset.train_data.numpy()
        # idx = (self.gen_dataset.svhn_dataset.train_labels==0) | (self.gen_dataset.svhn_dataset.train_labels==1) | (self.gen_dataset.svhn_dataset.train_labels==2)
        idx = (self.gen_dataset.svhn_dataset.train_labels == 0)
        self.gen_dataset.svhn_dataset.train_labels = np.tile(self.gen_dataset.svhn_dataset.train_labels[idx], 2)
        self.gen_dataset.svhn_dataset.train_data = np.tile(self.gen_dataset.svhn_dataset.train_data[idx], (2, 1, 1))
        self.gen_dataset.svhn_dataset.train_labels = torch.from_numpy(self.gen_dataset.svhn_dataset.train_labels)
        self.gen_dataset.svhn_dataset.train_data = torch.from_numpy(self.gen_dataset.svhn_dataset.train_data)
        print(len(self.train_dataset), len(self.test_dataset), len(self.gen_dataset))

        if self.params['sampling_dirichlet']:
            ## sample indices for participants using Dirichlet distribution
            indices_per_participant = self.sample_dirichlet_train_data(
                self.params['number_of_total_participants'],
                alpha=self.params['dirichlet_alpha'])
            train_loaders = [(pos, self.get_train(indices)) for pos, indices in
                             indices_per_participant.items()]
        else:
            ## sample indices for participants that are equally
            # splitted to 500 images per participant
            all_range = self.get_all_range_but_poison()
            random.shuffle(all_range)
            train_loaders = [(pos, self.get_train_old(all_range, pos))
                             for pos in range(self.params['number_of_total_participants'])]
        self.train_data = train_loaders
        self.test_data = self.get_test()
        self.poisoned_data_for_train = self.poison_dataset()
        print("Poison dataset len len len len :", len(self.poisoned_data_for_train))
        self.test_data_poison = self.poison_test_dataset()
        print("Poison test dataset len len len len :", len(self.test_data_poison))
        print("Training dataset len len len len :", len(self.train_data[0][1]))
        print("Test dataset len len len len :", len(self.test_data))
        self.poison_list = self.find_poison_idx()

    def find_poison_idx(self):
        p_list = []
        labels = np.array(self.train_dataset.svhn_dataset.train_labels)
        for pos, v in enumerate(labels):
            if v == 1:
                p_list.append(pos)
        print("P_List", len(p_list))
        return p_list

    def get_all_range_but_poison(self):
        labels = np.array(self.train_dataset.svhn_dataset.train_labels)
        range_no_id = list(range(len(labels)))
        # for pos, image in enumerate(labels):
        #     if image == 1:
        #         range_no_id.remove(pos)
        print("all range but 1", len(range_no_id))
        return range_no_id

    def get_train(self, indices):
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       indices), drop_last=True)
        return train_loader

    def get_train_old(self, all_range, model_no):
        data_len = int(len(all_range) / self.params['number_of_total_participants'])
        sub_indices = all_range[model_no * data_len: (model_no + 1) * data_len]
        print("data_len", data_len)
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       sub_indices), drop_last=True)
        return train_loader

    def get_secret_loader(self):
        """
        For poisoning we can use a larger data set. I don't sample randomly, though.

        """
        indices = list(range(len(self.train_dataset)))
        random.shuffle(indices)
        shuffled_indices = indices[:self.params['size_of_secret_dataset']]
        train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.params['batch_size'],
                                                   sampler=SubsetSampler(shuffled_indices))
        return train_loader

    def get_test(self):
        labels = np.array(self.test_dataset.svhn_dataset.test_labels)
        range_no_id = list(range(len(labels)))
        for pos, image in enumerate(labels):
            if image == 1:
                range_no_id.remove(pos)
        train_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                   batch_size=self.params['test_batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       range_no_id), drop_last=True)
        return train_loader

    def poison_dataset(self):
        indices = list()
        labels = np.array(self.train_dataset.svhn_dataset.train_labels)
        range_no_id = list(range(len(labels)))
        for pos, image in enumerate(labels):
            if image == 1:
                range_no_id.remove(pos)

        # add random images to other parts of the batch
        for batches in range(0, self.params['size_of_secret_dataset']):
            range_iter = random.sample(range_no_id, self.params['batch_size'])
            indices.extend(range_iter)

        return torch.utils.data.DataLoader(self.train_dataset,
                                           batch_size=self.params['batch_size'],
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(indices), drop_last=True)

    def poison_test_dataset(self):
        labels = np.array(self.test_dataset.svhn_dataset.test_labels)
        range_no_id = list(range(len(labels)))
        # for pos, image in enumerate(labels):
        #     if image != 2:
        #         range_no_id.remove(pos)
        train_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                   batch_size=self.params['test_batch_size'],
                                                   sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                       range_no_id), drop_last=True)
        return train_loader

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.cuda()
        target = target.cuda()
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target

    def _weights_init(self, module):
        '''
        Initializes weights for generator and discriminator
        :param module: generator or discriminator network expected
        '''
        classname = module.__class__.__name__
        if classname.find('Conv') != -1:
            module.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            module.weight.data.normal_(1.0, 0.02)
            module.bias.data.fill_(0)
