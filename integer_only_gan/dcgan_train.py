import argparse
import os
import numpy as np
import math
import sys

from diffaug import DiffAugment
policy = 'color,translation,cutout'

import torchvision.transforms as transforms
from torchvision.utils import save_image
from artbench import ArtBench10

from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm 

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
from torch.quantization.qconfig import QConfig
from torch.quantization import QuantStub, DeQuantStub

#from torch.ao.quantization.observer import (
#    MovingAverageMinMaxObserver,
#)
from torch.ao.quantization.fake_quantize import (
    FakeQuantize,
    #default_fused_wt_fake_quant,
)
from integer_only_observer import MovingAverageIntegerMinMaxObserver

from fid_score import calculate_fid_given_paths

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=400, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=16, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


HIDDEN_DIM = 1
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Linear(opt.latent_dim, HIDDEN_DIM * self.init_size ** 2)

        self.conv_blocks = nn.Sequential(
            #nn.BatchNorm2d(HIDDEN_DIM),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(1, 0.8),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, 3, stride=1, padding=1),
            nn.BatchNorm2d(HIDDEN_DIM, 0.8),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_DIM, opt.channels, 3, stride=1, padding=1),
        )

        self.tanh = nn.Tanh()

    def forward(self, z):
        z = self.quant(z)

        out = self.l1(z)
        out = out.view(out.shape[0], HIDDEN_DIM, self.init_size, self.init_size)
        img = self.conv_blocks(out)

        img = self.dequant(img)

        img = self.tanh(img)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

qconfig = QConfig(activation=FakeQuantize.with_args(
                            observer=MovingAverageIntegerMinMaxObserver,
                            quant_min=0,
                            quant_max=255,
                            reduce_range=True),
                  weight=FakeQuantize.with_args(
                            observer=MovingAverageIntegerMinMaxObserver,
                            quant_min=-128,
                            quant_max=127,
                            dtype=torch.qint8,
                            qscheme=torch.per_tensor_symmetric
                ))
generator.qconfig = qconfig

generator.eval()
    
# fuse the activations to preceding layers, where applicable
# this needs to be done manually depending on the model architecture
generator = torch.quantization.fuse_modules(generator,
   [["conv_blocks.1", "conv_blocks.2", "conv_blocks.3"]
   ,["conv_blocks.5", "conv_blocks.6", "conv_blocks.7"]]
)

generator.train()

# Prepare the model for QAT. This inserts observers and fake_quants in
# the model that will observe weight and activation tensors during calibration.
generator = torch.quantization.prepare_qat(generator)


# Configure data loader
os.makedirs("./data/artbench", exist_ok=True)
os.makedirs("./data/fid_eval", exist_ok=True)
os.makedirs("./data/fid_real", exist_ok=True)

dataloader = torch.utils.data.DataLoader(
    ArtBench10(
        "./data/artbench",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=opt.batch_size,
    shuffle=True,
)

testloader = torch.utils.data.DataLoader(
    ArtBench10(
        "./data/artbench",
        train=False,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),
    ),
    batch_size=1024,
    shuffle=False,
)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------
fid_every = 10

batches_done = 0
for epoch in range(opt.n_epochs):

    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Tensor(imgs.shape[0], 1).fill_(1.0)
        fake = Tensor(imgs.shape[0], 1).fill_(0.0)

        # Configure input
        real_imgs = imgs.type(Tensor)

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(DiffAugment(gen_imgs, policy=policy)), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(DiffAugment(real_imgs, policy=policy)), valid)
        fake_loss = adversarial_loss(discriminator(DiffAugment(gen_imgs, policy=policy).detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    if epoch % fid_every == fid_every - 1 or epoch == 0:
        with torch.no_grad():
            for i, (imgs, _) in tqdm(enumerate(testloader), total = len(testloader)):
                # Sample noise as generator input
                z = Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))

                # Generate a batch of images
                fake_imgs = generator(z)

                for j in range(imgs.shape[0]):
                    k = i * imgs.shape[0] + j
                    save_image(fake_imgs.data[j], "./data/fid_eval/%d.png" % k)
                    save_image(imgs.data[j], "./data/fid_real/%d.png" % k)

        fid = calculate_fid_given_paths(paths=('./data/fid_real', './data/fid_eval'), 
                batch_size=256, device='cuda', dims=2048)
        print(f"FID at {epoch}: {fid}")
        torch.save(generator.state_dict(), f'models/generator_{epoch}.pt')
        torch.save(discriminator.state_dict(), f'models/discriminator_{epoch}.pt')
        torch.save(generator.state_dict(), 'generator.pt')

torch.save(generator.cpu().state_dict(), 'generator.pt')
torch.save(discriminator.cpu().state_dict(), 'discriminator.pt')