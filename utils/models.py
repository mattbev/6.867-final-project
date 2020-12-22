import torch
from torch import nn

from utils.basics import initialize_weights

class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Linear(7*7*32, 10)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class AttackGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, input_size=28):
        super(AttackGenerator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU()
        )
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh()
        )

        initialize_weights(self)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv(x)
        return x



class UAP(nn.Module):
    def __init__(self, shape=(28, 28), num_channels=1, mean=[0.5], std=[0.5], use_cuda=False):
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
        orig_img = x * self.std_tensor + self.mean_tensor # Put image into original form
        adv_orig_img = orig_img + uap # Add uap to input
        adv_x = (adv_orig_img - self.mean_tensor)/self.std_tensor # Put image into normalized form
        return adv_x