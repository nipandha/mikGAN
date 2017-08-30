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


print("-- Part 1 -> DONE --")
print('')
# _______________________________________________________________________________________________________________________

# 2 - LOADING THE DATA
print("-- Part 1 -> DATA LOADING -")

print("-- Part 1 -> DONE --")
print('')
# _______________________________________________________________________________________________________________________

# 3 -


print('')
print("-- THE END -------------------------------------------------")
print('~~~ Have a nice day :) ~~~')
print('###########################################################################')
print('')