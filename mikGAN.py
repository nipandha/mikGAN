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
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torch.backends import cudnn
import torchvision
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
MODEL_PATH = './models'
SAMPLE_PATH = './samples'
IMAGE_PATH = './CelebA/128_crop'

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

# 3 - SOLVER
print("-- Part 3 -> SOLVER -")

class Solver(object):
    def __init__(self, data_loader):
        self.generator = None
        self.discriminator = None
        self.g_optimizer = None
        self.d_optimizer = None
        self.g_conv_dim = G_CONV_DIM
        self.d_conv_dim = D_CONV_DIM
        self.z_dim = Z_DIM
        self.momentum1 = MOMENTUM_1
        self.momentum2 = MOMENTUM_2
        self.image_size = IMAGE_SIZE
        self.data_loader = data_loader
        self.num_epochs = NUM_EPOCHS
        self.batch_size = BATCH_SIZE
        self.sample_size = SAMPLE_SIZE
        self.lr = LEARNING_RATE
        self.log_step = LOG_STEP
        self.sample_step = SAMPLE_STEP
        self.sample_path = SAMPLE_PATH
        self.model_path = MODEL_PATH
        self.build_model()

    def build_model(self):
        self.generator = Generator(z_dim=self.z_dim, image_size=self.image_size, conv_dim=self.g_conv_dim)
        self.discriminator = Discriminator(image_size=self.image_size, conv_dim=self.d_conv_dim)

        self.g_optimizer = optim.Adam(self.generator.parameters(), self.lr, [self.momentum1, self.momentum2])
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), self.lr, [self.momentum1, self.momentum2])

        if torch.cuda.is_available():
            self.generator.cuda()
            self.discriminator.cuda()

    def to_variable(self, x):
        """Conver tensor to Variable"""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)

    def to_data(self, x):
        """Convert Variable to tensor"""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def reset_grad(self):
        """Zero the gradients buffers"""
        self.discriminator.zero_grad()
        self.generator.zero_grad()

    def denorm(self, x):
        """Convert range (-1, 1) to (0, 1)"""
        out = (x + 1) / 2
        return out.clamp(0, 1)

    def train(self):
        """Train generator and discriminator"""
        fixed_noise = self.to_variable(torch.randn(self.batch_size, self.z_dim))
        total_step = len(self.data_loader)
        print("Training...")
        for epoch in range(self.num_epochs):
            for i, images in enumerate(self.data_loader):

                # ===== TRAIN D: =====
                images = self.to_variable(images)
                batch_size = images.size(0)
                noise = self.to_variable(torch.randn(batch_size, self.z_dim))

                    # Train D to recognize real images as REAL
                outputs = self.discriminator(images)
                real_loss = torch.mean((outputs - 1) ** 2)

                    # Train D to recognize fake images as FAKE
                fake_images = self.generator(noise)
                outputs = self.discriminator(fake_images)
                fake_loss = torch.mean(outputs ** 2)

                    # Backprop + Optimize
                d_loss = real_loss + fake_loss
                self.reset_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # ===== TRAIN G =====
                noise = self.to_variable(torch.randn(batch_size, self.z_dim))

                    # Train G so that D recognizes G(z) as real
                fake_images = self.generator(noise)
                outputs = self.discriminator(fake_images)
                g_loss = torch.mean((outputs-1) ** 2)

                    # Backprop & Optimize
                self.reset_grad()
                g_loss.backward()
                self.g_optimizer.step()

                # PRINT INFO:
                if (i+1) % self.log_step == 0:
                    print('Epoch [%d/%d], Step[%d/%d], d_real_loss: %.4f, d_fake_loss: %.4f, g_loss: %.4f' %
                          (epoch+1, self.num_epochs, i+1, total_step, real_loss.data[0], fake_loss.data[0], g_loss.data[0]))

                # SAVE SAMPLED IMAGES:
                if (i+1) % self.sample_step == 0:
                    fake_images =self.generator(fixed_noise)
                    torchvision.utils.save_image(self.denorm(fake_images.data), os.path.join(self.sample_path,
                                                                                             'fake_samples-%d-%d.png' %
                                                                                             (epoch+1, i+1)))
            # Save the model and parameters for each epoch
            g_path = os.path.join(self.model_path, 'generator-%d.pkl' % (epoch + 1))
            d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' % (epoch + 1))
            torch.save(self.generator.state_dict(), g_path)
            torch.save(self.discriminator.state_dict(), d_path)

    def sample(self):
        # Load trained parameters
        g_path = os.path.join(self.model_path, 'generator-%d.pkl' % (self.num_epochs))
        d_path = os.path.join(self.model_path, 'discriminator-%d.pkl' % (self.num_epochs))
        self.generator.load_state_dict(torch.load(g_path))
        self.discriminator.load_state_dict(torch.load(d_path))
        self.generator.eval()
        self.discriminator.eval()

        # Sample images
        noise = self.to_variable(torch.randn(self.sample_size, self.z_dim))
        fake_images = self.generator(noise)
        sample_path = os.path.join(self.sample_path, 'fake_samples-final.png')
        torchvision.utils.save_image(self.denorm(fake_images.data), sample_path, nrow=12)

        print("Saved sampled images to '%s'" % sample_path)



print("-- Part 3 -> DONE --")
print('')
# _______________________________________________________________________________________________________________________

# 4 - DO THE WORK
print("-- Part 4 -> WORK START -")

cudnn.benchmark = True

data_loader = get_loader(IMAGE_PATH, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS)

solver = Solver(data_loader=data_loader)

# Create directories if not exist
if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
if not os.path.exists(SAMPLE_PATH):
    os.makedirs(SAMPLE_PATH)

# Train ans Sample
solver.train()

solver.sample()

print('')
print("-- THE END -------------------------------------------------")
print('~~~ Have a nice day :) ~~~')
print('###########################################################################')
print('')