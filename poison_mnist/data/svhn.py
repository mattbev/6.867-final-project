'''
Defines the dataset and loader for the SVHN images
'''
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class SvhnDataset(Dataset):
    '''
    The dataset class is, effectively, an iterator over the data.
    As training data we have images, labels and masks to use only a limited number of labels.
    As test data we have images and labels
    '''

    def __init__(self, image_size, split, dir_root, use_gpu):
        '''
        :param image_size: required image size. The images are resized to match the size
        :param split: defines train or test split
        :param dir_root: directory root to store loaded data
        :param use_gpu: indication to use the GPU
        '''
        self.split = split
        self.dir_root = dir_root
        self.use_gpu = use_gpu
        self.svhn_dataset = self._create_dataset(image_size)
        self.label_mask = self._create_label_mask()

    def _create_dataset(self, image_size):
        '''
        Loads the dataset, normalizes and resizes images

        :param image_size: required image size
        '''
        normalize = transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5])
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize])
        # return datasets.SVHN(root=self.dir_root, download=True,transform=transform, split=self.split)
        # return datasets.CIFAR10(self.dir_root, train=True if self.split=="train" else False, download=True,transform=transform)
        return datasets.MNIST(self.dir_root, train=True if self.split == "train" else False, download=True,
                              transform=transform)

    def _is_train_dataset(self):
        '''
        Returns True if this is a train dataset, False - otherwise
        '''
        return True if self.split == 'train' else False

    def _create_label_mask(self):
        '''
        Creates a mask array to use only a limited number of labels during the training
        '''
        if self._is_train_dataset():
            label_mask = np.zeros(len(self.svhn_dataset))
            label_mask[0:1000] = 1
            np.random.shuffle(label_mask)
            label_mask = torch.LongTensor(label_mask)
            return label_mask
        return None

    def __len__(self):
        return len(self.svhn_dataset)

    def __getitem__(self, idx):
        data, label = self.svhn_dataset.__getitem__(idx)
        # if self._is_train_dataset():
        #     return data, label, self.label_mask[idx]
        return data, label


def create_loaders(image_size, batch_size, dir_root, num_workers):
    '''
    Creates loaders for train and test datasets
    '''
    use_gpu = True if torch.cuda.is_available() else False

    svhn_train = SvhnDataset(image_size, 'train', dir_root, use_gpu)
    svhn_test = SvhnDataset(image_size, 'test', dir_root, use_gpu)

    svhn_loader_train = DataLoader(
        dataset=svhn_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    svhn_loader_test = DataLoader(
        dataset=svhn_test,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return svhn_loader_train, svhn_loader_test
