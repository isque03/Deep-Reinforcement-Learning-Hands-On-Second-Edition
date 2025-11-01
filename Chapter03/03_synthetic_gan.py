#!/usr/bin/env python
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter

import torchvision.utils as vutils

import numpy as np

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

LATENT_VECTOR_SIZE = 100
DISCR_FILTERS = 64
GENER_FILTERS = 64
BATCH_SIZE = 16

# dimension input image will be rescaled
IMAGE_SIZE = 64

LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=DISCR_FILTERS,
                      kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS, out_channels=DISCR_FILTERS*2,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS*2),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 2, out_channels=DISCR_FILTERS * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 4, out_channels=DISCR_FILTERS * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(DISCR_FILTERS * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=DISCR_FILTERS * 8, out_channels=1,
                      kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.view(-1, 1).squeeze(dim=1)


class Generator(nn.Module):
    def __init__(self, output_shape):
        super(Generator, self).__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_VECTOR_SIZE, out_channels=GENER_FILTERS * 8,
                               kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(GENER_FILTERS * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 8, out_channels=GENER_FILTERS * 4,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 4, out_channels=GENER_FILTERS * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS * 2, out_channels=GENER_FILTERS,
                               kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(GENER_FILTERS),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=GENER_FILTERS, out_channels=output_shape[0],
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


def generate_real_batch(batch_size=BATCH_SIZE):
    """Generate synthetic 'real' images - simple patterns"""
    batch = []
    for _ in range(batch_size):
        # Create a simple pattern image
        img = np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        
        # Add some random patterns
        for i in range(3):
            # Random rectangles
            x1, y1 = np.random.randint(0, IMAGE_SIZE//2), np.random.randint(0, IMAGE_SIZE//2)
            x2, y2 = x1 + np.random.randint(10, IMAGE_SIZE//2), y1 + np.random.randint(10, IMAGE_SIZE//2)
            img[i, y1:y2, x1:x2] = np.random.uniform(0.5, 1.0)
            
            # Random circles
            center_x, center_y = np.random.randint(IMAGE_SIZE//4, 3*IMAGE_SIZE//4), np.random.randint(IMAGE_SIZE//4, 3*IMAGE_SIZE//4)
            radius = np.random.randint(5, IMAGE_SIZE//8)
            y, x = np.ogrid[:IMAGE_SIZE, :IMAGE_SIZE]
            mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
            img[i][mask] = np.random.uniform(0.3, 0.8)
        
        batch.append(img)
    
    return torch.tensor(np.array(batch), dtype=torch.float32)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action='store_true',
        help="Enable cuda computation")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    
    # Define image shape (3 channels, 64x64)
    input_shape = (3, IMAGE_SIZE, IMAGE_SIZE)

    net_discr = Discriminator(input_shape=input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(
        params=net_gener.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(
        params=net_discr.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    writer = SummaryWriter()

    gen_losses = []
    dis_losses = []
    iter_no = 0

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    print("Starting GAN training with synthetic images...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            # Generate real batch
            batch_v = generate_real_batch().to(device)
            
            # fake samples, input is 4D: batch, filters, x, y
            gen_input_v = torch.FloatTensor(
                BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
            gen_input_v.normal_(0, 1)
            gen_input_v = gen_input_v.to(device)
            gen_output_v = net_gener(gen_input_v)

            # train discriminator
            dis_optimizer.zero_grad()
            dis_output_true_v = net_discr(batch_v)
            dis_output_fake_v = net_discr(gen_output_v.detach())
            dis_loss = objective(dis_output_true_v, true_labels_v) + \
                       objective(dis_output_fake_v, fake_labels_v)
            dis_loss.backward()
            dis_optimizer.step()
            dis_losses.append(dis_loss.item())

            # train generator
            gen_optimizer.zero_grad()
            dis_output_v = net_discr(gen_output_v)
            gen_loss_v = objective(dis_output_v, true_labels_v)
            gen_loss_v.backward()
            gen_optimizer.step()
            gen_losses.append(gen_loss_v.item())

            iter_no += 1
            if iter_no % REPORT_EVERY_ITER == 0:
                print("Iter %d: gen_loss=%.3e, dis_loss=%.3e" % (
                    iter_no, np.mean(gen_losses), np.mean(dis_losses)))
                writer.add_scalar(
                    "gen_loss", np.mean(gen_losses), iter_no)
                writer.add_scalar(
                    "dis_loss", np.mean(dis_losses), iter_no)
                gen_losses = []
                dis_losses = []
            if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
                writer.add_image("fake", vutils.make_grid(
                    gen_output_v.data[:64], normalize=True), iter_no)
                writer.add_image("real", vutils.make_grid(
                    batch_v.data[:64], normalize=True), iter_no)
                print(f"Saved images at iteration {iter_no}")
    except KeyboardInterrupt:
        print("\nTraining stopped by user")
        writer.close()
