#!/usr/bin/env python
import random
import argparse
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import torchvision.utils as vutils

import ale_py  # This registers ALE environments
import gymnasium as gym
import gymnasium.spaces

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
SAVE_MODEL_EVERY_ITER = 100  # Save every 100 iterations for frequent checkpoints


class InputWrapper(gym.ObservationWrapper):
    """
    Preprocessing of input numpy array:
    1. resize image into predefined size
    2. move color channel axis to a first place
    """
    def __init__(self, *args):
        super(InputWrapper, self).__init__(*args)
        assert isinstance(self.observation_space, gymnasium.spaces.Box)
        old_space = self.observation_space
        self.observation_space = gymnasium.spaces.Box(
            self.observation(old_space.low),
            self.observation(old_space.high),
            dtype=np.float32)

    def observation(self, observation):
        # resize image
        new_obs = cv2.resize(
            observation, (IMAGE_SIZE, IMAGE_SIZE))
        # transform (210, 160, 3) -> (3, 210, 160)
        new_obs = np.moveaxis(new_obs, 2, 0)
        return new_obs.astype(np.float32)


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


def iterate_batches(envs, batch_size=BATCH_SIZE):
    batch = []
    env_gen = iter(lambda: random.choice(envs), None)

    while True:
        e = next(env_gen)
        # Reset environment if needed
        if not hasattr(e, '_reset_called') or not e._reset_called:
            obs, info = e.reset()
            e._reset_called = True
        else:
            obs, reward, terminated, truncated, info = e.step(e.action_space.sample())
            if np.mean(obs) > 0.01:
                batch.append(obs)
            if terminated or truncated:
                obs, info = e.reset()
                e._reset_called = True
        
        if len(batch) == batch_size:
            # Normalising input between -1 to 1
            batch_np = np.array(batch, dtype=np.float32) * 2.0 / 255.0 - 1.0
            yield torch.tensor(batch_np)
            batch.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", default=False, action='store_true',
        help="Enable cuda computation")
    parser.add_argument(
        "--resume", default=None, type=str,
        help="Resume from checkpoint file (e.g., checkpoint_iter_100.pth)")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")
    envs = [
        InputWrapper(gym.make(name, render_mode='rgb_array'))
        for name in ('ALE/Breakout-v5', 'ALE/AirRaid-v5', 'ALE/Pong-v5')
    ]
    
    # Initialize all environments
    for env in envs:
        env.reset()
        env._reset_called = True
    input_shape = envs[0].observation_space.shape

    net_discr = Discriminator(input_shape=input_shape).to(device)
    net_gener = Generator(output_shape=input_shape).to(device)

    objective = nn.BCELoss()
    gen_optimizer = optim.Adam(
        params=net_gener.parameters(), lr=LEARNING_RATE,
        betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(
        params=net_discr.parameters(), lr=LEARNING_RATE * 0.5,
        betas=(0.5, 0.999))
    writer = SummaryWriter(log_dir='runs/atari_gan')

    gen_losses = []
    dis_losses = []
    iter_no = 0
    
    # Track best performance for smart checkpointing
    best_gen_loss = float('inf')
    best_dis_accuracy = 0.0
    best_checkpoint_iter = 0
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"🔄 Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        net_gener.load_state_dict(checkpoint['generator_state_dict'])
        net_discr.load_state_dict(checkpoint['discriminator_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
        iter_no = checkpoint['iter_no']
        print(f"✅ Resumed from iteration {iter_no}")
    else:
        print("🚀 Starting fresh training")

    true_labels_v = torch.ones(BATCH_SIZE, device=device)
    fake_labels_v = torch.zeros(BATCH_SIZE, device=device)

    for batch_v in iterate_batches(envs):
        # fake samples, input is 4D: batch, filters, x, y
        gen_input_v = torch.FloatTensor(
            BATCH_SIZE, LATENT_VECTOR_SIZE, 1, 1)
        gen_input_v.normal_(0, 1)
        gen_input_v = gen_input_v.to(device)
        batch_v = batch_v.to(device)
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
            # Calculate additional metrics for collapse detection
            gen_loss_mean = np.mean(gen_losses)
            dis_loss_mean = np.mean(dis_losses)
            
            # Discriminator accuracy (how well it distinguishes real vs fake)
            dis_accuracy_real = (dis_output_true_v > 0.5).float().mean().item()
            dis_accuracy_fake = (dis_output_fake_v < 0.5).float().mean().item()
            dis_accuracy = (dis_accuracy_real + dis_accuracy_fake) / 2
            
            # Generator confidence (how confident it is in fooling discriminator)
            gen_confidence = dis_output_v.mean().item()
            
            # Loss ratio (indicator of training balance)
            loss_ratio = dis_loss_mean / (gen_loss_mean + 1e-8)
            
            # Image diversity metrics
            fake_std = gen_output_v.std().item()
            fake_mean = gen_output_v.mean().item()
            
            log.info("Iter %d: gen_loss=%.3e, dis_loss=%.3e, dis_acc=%.3f, gen_conf=%.3f, loss_ratio=%.3f",
                     iter_no, gen_loss_mean, dis_loss_mean, dis_accuracy, gen_confidence, loss_ratio)
            
            # Log all metrics to TensorBoard
            writer.add_scalar("Losses/gen_loss", gen_loss_mean, iter_no)
            writer.add_scalar("Losses/dis_loss", dis_loss_mean, iter_no)
            writer.add_scalar("Losses/loss_ratio", loss_ratio, iter_no)
            
            writer.add_scalar("Accuracy/dis_accuracy", dis_accuracy, iter_no)
            writer.add_scalar("Accuracy/dis_accuracy_real", dis_accuracy_real, iter_no)
            writer.add_scalar("Accuracy/dis_accuracy_fake", dis_accuracy_fake, iter_no)
            
            writer.add_scalar("Generator/confidence", gen_confidence, iter_no)
            writer.add_scalar("Generator/output_std", fake_std, iter_no)
            writer.add_scalar("Generator/output_mean", fake_mean, iter_no)
            
            # Mode collapse detection
            if dis_accuracy > 0.95:
                log.warning("⚠️  High discriminator accuracy (%.3f) - possible mode collapse!", dis_accuracy)
            if loss_ratio > 10:
                log.warning("⚠️  High loss ratio (%.3f) - discriminator too strong!", loss_ratio)
            if fake_std < 0.1:
                log.warning("⚠️  Low generator diversity (%.3f) - possible mode collapse!", fake_std)
            
            gen_losses = []
            dis_losses = []
        if iter_no % SAVE_IMAGE_EVERY_ITER == 0:
            writer.add_image("fake", vutils.make_grid(
                gen_output_v.data[:64], normalize=True), iter_no)
            writer.add_image("real", vutils.make_grid(
                batch_v.data[:64], normalize=True), iter_no)
        
        # Smart checkpointing - only save when performance improves
        if iter_no % SAVE_MODEL_EVERY_ITER == 0:
            # Calculate current performance metrics
            current_gen_loss = gen_loss_mean
            current_dis_accuracy = dis_accuracy
            
            # Check if this is the best performance so far
            # We want lower generator loss and balanced discriminator accuracy (not too high, not too low)
            is_improvement = False
            
            # Generator improvement: lower loss is better
            if current_gen_loss < best_gen_loss:
                is_improvement = True
                best_gen_loss = current_gen_loss
                log.info("🎯 New best generator loss: %.3f (was %.3f)", current_gen_loss, best_gen_loss)
            
            # Discriminator balance: we want accuracy around 0.7-0.8 (not too high = mode collapse)
            if 0.6 <= current_dis_accuracy <= 0.9 and abs(current_dis_accuracy - 0.75) < abs(best_dis_accuracy - 0.75):
                is_improvement = True
                best_dis_accuracy = current_dis_accuracy
                log.info("🎯 New best discriminator balance: %.3f (was %.3f)", current_dis_accuracy, best_dis_accuracy)
            
            # Save checkpoint if improved
            if is_improvement:
                checkpoint = {
                    'iter_no': iter_no,
                    'generator_state_dict': net_gener.state_dict(),
                    'discriminator_state_dict': net_discr.state_dict(),
                    'gen_optimizer_state_dict': gen_optimizer.state_dict(),
                    'dis_optimizer_state_dict': dis_optimizer.state_dict(),
                    'gen_loss': current_gen_loss,
                    'dis_accuracy': current_dis_accuracy,
                }
                torch.save(checkpoint, f'checkpoint_iter_{iter_no}.pth')
                best_checkpoint_iter = iter_no
                log.info("💾 Checkpoint saved at iteration %d (gen_loss=%.3f, dis_acc=%.3f)", 
                        iter_no, current_gen_loss, current_dis_accuracy)
            else:
                log.info("⏭️  No improvement at iteration %d (gen_loss=%.3f, dis_acc=%.3f) - skipping checkpoint", 
                        iter_no, current_gen_loss, current_dis_accuracy)
    
    # Save final checkpoint when training ends
    log.info("🏁 Training completed. Best checkpoint was at iteration %d", best_checkpoint_iter)
