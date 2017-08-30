#################################################################################################
"""
                    MIK_GAN

                    Deep Convolutive Generating Adversarial Network using Pytorch

                    Implemented following the implementation by Yunjey found in:
                        https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/deep_convolutional_gan


"""
##################################################################################################

# How it works:
    # Model
        # Discriminator: Binary classifier -> classify real & fake images
            # 1 <=> Real image
            # 0 <=> Fake image
        # Generator: Generative model -> Creates image form code
            # Generates an image that can not be distinguishable form the real image

# 0 - IMPORTS:

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils import data
from torchvision import transforms


import os

from PIL import Image

# 00 - CONSTANTS:

IMAGE_SIZE = 64 # Default=64
Z_DIM = 100 # Default=100
G_CONV_DIM = 64 # Default=64
D_CONV_DIM = 64 # Default=64

# Training:
NUM_EPOCHS = 20 # Default=20
BATCH_SIZE = 32 # Default=32
SAMPLE_SIZE = 100 # Default=100
NUM_WORKERS = 2 # Default=2
LEARNING_RATE = 0.0002 # Default=0.0002
MOMENTUM_1 = 0.5 # Default=0.5
MOMENTUM_2 = 0.999 # Default=0.999

#Paths:
MODEL_PATH =
SAMPLE_PATH =
IMAGE_PATH =

LOG_STEP = 10 # Default=10
SAMPLE_STEP = 500 # Default=500
# _______________________________________________________________________________________________________________________

# 1 - LOADING THE DATA
print("-- Part 1 -> DATA LOADING -")

class ImageFolder(data.Dataset):

    def __inti__(self, root, transform=None):
        """Initializes images paths and preprocessing module."""
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path= self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """Returns the total amount of images"""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2):
    """Builds and returns a Dataloader"""

    transform = transforms.Compose([
        transforms.Scale(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImageFolder(image_path, transform)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return data_loader





print("-- Part 1 -> DONE --")
print('')
# _______________________________________________________________________________________________________________________

# 2 - MODEL DEFINE:
print("-- Part 2 -> MODEL DEFINE -")


# 2.1 - GENERATOR:
def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity"""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Generator(nn.Module):
    """Generator containig 7 deconvolutional layers"""
    def _init__(self, z_dim=256, image_size=128, conv_dim=64):
        super(Generator, self).__init__()
        self.fc = deconv(z_dim, conv_dim*8, int(image_size/16), 1, 0, bn=False)
        self.deconv1 = deconv(conv_dim * 8, conv_dim * 4, 4)
        self.deconv2 = deconv(conv_dim * 4, conv_dim * 2, 4)
        self.deconv3 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim, 3, 4, bn=False)

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        out = self.fc(z)
        out = F.leaky_relu(self.deconv1(out), 0.05)
        out = F.leaky_relu(self.deconv2(out), 0.05)
        out = F.leaky_relu(self.deconv3(out), 0.05)
        out = F.tanh(self.deconv4(out))
        return out


# 2.2 - DISCRIMINATOR:
def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity"""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class Discriminator(nn.Module):
    """Discriminator conaining 4 convolutional layers"""
    def __init__(self, image_size=128, conv_dim=64):
        super(Discriminator, self).__init__()
        self.conv1 = conv(3, conv_dim, 4, bn=False)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4)
        self.conv3 = conv(conv_dim * 2, conv_dim * 4, 4)
        self.conv4 = conv(conv_dim * 4, conv_dim * 8, 4)
        self.fc = conv(conv_dim * 8, 1, int(image_size/16), 1, 0, False)

    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)
        out = F.leaky_relu(self.conv2(out), 0.05)
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = self.fc(out).squeeze()
        return out

print("-- Part 2 -> DONE --")
print('')
# _______________________________________________________________________________________________________________________

# 3 -


print('')
print("-- THE END -------------------------------------------------")
print('~~~ Have a nice day :) ~~~')
print('###########################################################################')
print('')