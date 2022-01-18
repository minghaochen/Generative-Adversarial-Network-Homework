import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(level=logging.INFO,
                    filename='log/train_state.log',
                    filemode='a',
                    format='%(message)s'
                    )


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adamw: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adamw: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=200, help="interval betwen image samples")
parser.add_argument("--d_times", type=int, default=1, help="Learning D: Repeat d_times")
opt = parser.parse_args()
img_shape = (opt.channels, opt.img_size, opt.img_size)

# Configure data loader
train_data_path = '../../data/AnimeDataset/'
train_set = datasets.ImageFolder(root=train_data_path,
                                 transform=transforms.Compose(
                                     [transforms.Resize(opt.img_size),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5], [0.5])] # normalize the images between -1 and 1
                                 )
                                 )
dataloader = torch.utils.data.DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        output = self.model(input)
        return output

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):

        output = self.model(input)

        return output.view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

# Initialize generator and discriminator
generator = Generator(nz=100, ngf=64, nc=3).to(device)
generator.apply(weights_init)
discriminator = Discriminator(nc=3, ndf=64).to(device)
discriminator.apply(weights_init)
# Loss function
loss_fcn = torch.nn.BCELoss()
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
Tensor = torch.FloatTensor
for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        imgs = imgs.to(device)
        valid = Variable(Tensor(imgs.size(0)).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(Tensor(imgs.size(0)).fill_(0.0), requires_grad=False).to(device)
        # Configure input
        real_imgs = Variable(imgs.type(Tensor)).to(device)
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).view(imgs.shape[0], opt.latent_dim,1,1).to(device)
        # Generate a batch of images
        gen_imgs = generator(z)

        for k in range(opt.d_times):
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss_fcn(discriminator(real_imgs), valid)
            fake_loss = loss_fcn(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        optimizer_G.zero_grad()
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).view(imgs.shape[0], opt.latent_dim,1,1).to(device)
        # Generate a batch of images
        gen_imgs = generator(z)
        # Loss measures generator's ability to fool the discriminator
        g_loss = loss_fcn(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        print("[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]".format(epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        logging.info("[Epoch {}/{}] [Batch {}/{}] [D loss: {:.4f}] [G loss: {:.4f}]".format(epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item(), ))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "log/images/%d.png" % batches_done, nrow=5, normalize=True)
