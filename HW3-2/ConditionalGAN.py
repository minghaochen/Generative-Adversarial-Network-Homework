import pandas as pd
import numpy as np
import torch
import os
import PIL.Image as Image
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image
import logging


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyDataSet(Dataset):
    def __init__(self, images_path, tags_path):
        self.images_path = images_path
        self.tags_path = tags_path

        df_tags = pd.read_csv(tags_path, names=["image", "tag"])
        df_tags['hair'] = df_tags.tag.apply(lambda x: x.split(' ')[0])
        df_tags['eye'] = df_tags.tag.apply(lambda x: x.split(' ')[2])
        self.images = [os.path.join(self.images_path, str(x) + '.jpg') for x in df_tags['image']]

        hair_dict = df_tags['hair'].drop_duplicates().values
        eye_dict = df_tags['eye'].drop_duplicates().values

        self.hair_num = len(hair_dict)
        self.eye_num = len(eye_dict)

        index = []
        for i in range(max(len(hair_dict), len(eye_dict))):
            index.append(str(i + 1))

        self.hair_dict = dict(zip(hair_dict, index))
        self.eye_dict = dict(zip(eye_dict, index))

        self.hairs = [self.hair_dict[x] for x in df_tags['hair']]
        self.eyes = [self.eye_dict[x] for x in df_tags['eye']]

        self.inv_hair_dict = {v:k for k,v in self.hair_dict.items()}
        self.inv_eye_dict = {v: k for k, v in self.eye_dict.items()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = Image.open(self.images[idx])
        hair = self.hairs[idx]
        eye = self.eyes[idx]
        return T.Compose([T.Resize(opt.img_size),T.ToTensor(),T.Normalize([0.5], [0.5])])(img), int(hair), int(eye)

images_path = '../../data/extra_data/images'
tags_path = '../../data/extra_data/tags.csv'
dataset = MyDataSet(images_path, tags_path)
HAIR_NUM = dataset.hair_num
EYE_NUM = dataset.eye_num
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)


class ModelD(nn.Module):

    def __init__(self):
        super(ModelD, self).__init__()
        #1. 文本需要的网络层
        self.txt_dense = nn.Linear(22,256)
        #2. 图像需要的网络层
        # self.con1 = nn.Conv2d(3,64,5,stride=2)
        nc = 3
        ndf = 64
        self.con1 = nn.Sequential(
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
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.output = nn.Sequential(
            nn.Conv2d(512+256,512,4,2,1),
            nn.Flatten(),
            nn.Linear(512*2*2,1)
        )
    def forward(self, x, labels):
        #1. 文本处理
        labels = F.relu(self.txt_dense(labels)).reshape([-1,256,1,1])  #(bs,256,1,1)
        # labels = torch.tile(labels,(1,1,4,4))  #(bs,256,4,4)
        labels = labels.repeat(1,1,4,4)
        #2. 处理图像
        x = self.con1(x) #(batch_size,512,4,4)
        #3. 连接图像和文本
        x = torch.cat((x,labels),axis=1)  #(bs,512+256,4,4)
        x = self.output(x)
        # return x
        return F.sigmoid(x)


class ModelG(nn.Module):

    def __init__(self, z_dim):
        super(ModelG, self).__init__()

        self.z_dim = z_dim
        #1. 文本需要的网络层
        self.text_layer = nn.Linear(22,256)
        #2. 图像需要的网络层
        self.img_layer = nn.Linear(356,512*4*4)
        self.img_bn = nn.BatchNorm2d(512)
        self.img_con = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.output = nn.ConvTranspose2d(64,3,4,stride=2,padding=1)

    def forward(self, x, labels):
        #1. 噪声和文本处理
        labels = F.relu(self.text_layer(labels))  #(bs,256)
        x = torch.cat((x,labels),axis=1)  #(bs,356)
        #2. 变为图像格式
        x = self.img_layer(x).reshape([-1,512,4,4])
        x = F.relu(self.img_bn(x))
        #3. 转置卷积
        x = self.img_con(x)  #(bs,64,32,32)
        x = self.output(x) #(bs,3,64,64)
        return F.tanh(x)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)

model_d = ModelD().to(device)
model_g = ModelG(z_dim=100).to(device)
model_d.apply(weights_init)
print(model_d)
model_g.apply(weights_init)
print(model_g)
# 测试输出尺寸
# out = model_g(torch.ones([1, 100]).to(device),torch.ones([1, 22]).to(device))
# print('生成器输出尺寸：', out.shape)
#
# out = model_d(torch.ones([1, 3, 64, 64]).to(device),torch.ones([1, 22]).to(device))
# print('判别器输出尺寸：', out.shape)


loss_fcn = torch.nn.BCELoss()
# Optimizers
optimizer_G = torch.optim.Adam(model_g.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.SGD(model_d.parameters(), lr=opt.lr)



Tensor = torch.FloatTensor

for epoch in range(opt.n_epochs):
    for i, (imgs, hair, eye) in enumerate(dataloader):
        imgs = imgs.to(device)

        hair = F.one_hot(hair-1, num_classes=HAIR_NUM)
        eye = F.one_hot(eye-1, num_classes=EYE_NUM)
        real_text = torch.cat((hair, eye), axis=1).float().to(device)

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False).to(device)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False).to(device)
        # Use Soft and Noisy Labels
        # valid = Variable(Tensor(np.random.uniform(0.7, 1.2, size=(imgs.size(0), 1))), requires_grad=False).to(device)
        # fake = Variable(Tensor(np.random.uniform(0.0, 0.3, size=(imgs.size(0), 1))), requires_grad=False).to(device)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor)).to(device)
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(device)
        rand_hair = torch.from_numpy(np.random.randint(0, HAIR_NUM, size=(64, 1)).astype(np.int64))
        rand_eye = torch.from_numpy(np.random.randint(0, EYE_NUM, size=(64, 1)).astype(np.int64))
        one_hot_hair = F.one_hot(rand_hair, num_classes=HAIR_NUM).reshape(shape=[64, HAIR_NUM])
        one_hot_eye = F.one_hot(rand_eye, num_classes=EYE_NUM).reshape(shape=[64, EYE_NUM])
        fake_text = torch.cat((one_hot_hair, one_hot_eye), axis=1).float().to(device)

        # Generate a batch of images
        gen_imgs = model_g(z, fake_text)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        for k in range(opt.d_times):
            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = loss_fcn(model_d(real_imgs, real_text), valid)

            fake_loss = loss_fcn(model_d(gen_imgs.detach(), fake_text), fake)
            d_loss = real_loss + fake_loss

            d_loss.backward()
            optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()
        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(device)
        rand_hair = torch.from_numpy(np.random.randint(0, HAIR_NUM, size=(64, 1)).astype(np.int64))
        rand_eye = torch.from_numpy(np.random.randint(0, EYE_NUM, size=(64, 1)).astype(np.int64))
        one_hot_hair = F.one_hot(rand_hair, num_classes=HAIR_NUM).reshape(shape=[64, HAIR_NUM])
        one_hot_eye = F.one_hot(rand_eye, num_classes=EYE_NUM).reshape(shape=[64, EYE_NUM])
        fake_text = torch.cat((one_hot_hair, one_hot_eye), axis=1).float().to(device)
        # Generate a batch of images
        gen_imgs = model_g(z, fake_text)
        # Loss measures generator's ability to fool the discriminator
        g_loss = loss_fcn(model_d(gen_imgs, fake_text), valid)

        g_loss.backward()
        optimizer_G.step()

        print("[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]".format(epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))
        logging.info("[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G loss: {}]".format(epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            hair = (fake_text[0].cpu() == 1).nonzero(as_tuple=True)[0].numpy()[0] + 1
            eye = (fake_text[0].cpu() == 1).nonzero(as_tuple=True)[0].numpy()[1] - 12 + 1
            hair = dataset.inv_hair_dict[str(hair)]
            eye = dataset.inv_eye_dict[str(eye)]
            save_image(gen_imgs.data[0], f"log/images/{hair} hair {eye} eye.png", nrow=1, normalize=True)

