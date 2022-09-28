import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)
logging.disable(logging.INFO)


# Define the model
class Diffusion:
    def __init__(
        self,
        noise_steps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        img_size=64,
        device="cuda",
    ):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(self.device)
        self.alpha = 1 - self.beta
        self.alpha_hat = self.alpha.cumprod(dim=0)

    def prepare_noise_schedule(self):
        beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        return beta

    def forward_diffusion(self, x, t):  # x: (n, 3, 64, 64), t: (n)
        logging.info(f"Forward diffusion: adding noises to {x.shape[0]} images")
        noise = torch.randn_like(x)  # noisy image
        x_t = (
            torch.sqrt(self.alpha_hat[t][:, None, None, None]) * x
            + torch.sqrt(1 - self.alpha_hat[t][:, None, None, None]) * noise
        )
        return x_t, noise

    def sample_timesteps(self, n):
        return torch.randint(1, self.noise_steps, (n,), device=self.device)

    def backward_diffusion(self, model, n):  # unnormalized output
        logging.info(f"Backward diffusion: sampling {n} images from diffusion model")
        model.eval()
        with torch.no_grad():
            x = torch.randn(n, 3, self.img_size, self.img_size).to(self.device)
            for i in tqdm(
                reversed(range(1, self.noise_steps)), ncols=70
            ):  # backward diffusion
                t = (torch.ones(n) * i).long().to(self.device)
                noise_pred = model(x, t)
                if i > 1:
                    noise = torch.randn_like(x)  # noisy image
                else:
                    noise = torch.zeros_like(x)
                x -= (
                    self.beta[t][:, None, None, None]
                    / torch.sqrt(1 - self.alpha_hat[t][:, None, None, None])
                    * noise_pred
                )
                x /= torch.sqrt(self.alpha[t][:, None, None, None])
                x += torch.sqrt(self.beta[t][:, None, None, None]) * noise
        model.train()
        return x


class UNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, time_dim=256, img_size=64, device="cuda"
    ):
        super().__init__()
        self.device = device
        self.time_dim = time_dim

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, img_size // 2)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, img_size // 4)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, img_size // 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, img_size // 4)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, img_size // 2)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, img_size)
        self.outc = nn.Conv2d(64, out_channels, 1)

    def pos_encoding(self, t, channels):  # t: (n, 1), channels: (1)
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)  # (n, channels)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(1).type(torch.float)  # (n, 1)
        t = self.pos_encoding(t, self.time_dim)  # (n, time_dim)

        x1 = self.inc(x)
        x2 = self.sa1(self.down1(x1, t))
        x3 = self.sa2(self.down2(x2, t))
        x4 = self.sa3(self.down3(x3, t))

        x4 = self.bot3(self.bot2(self.bot1(x4)))

        x = self.sa4(self.up1(x4, x3, t))
        x = self.sa5(self.up2(x, x2, t))
        x = self.sa6(self.up3(x, x1, t))
        x = self.outc(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.conv(x))
        else:
            return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )
        self.embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        x = x + self.embed(t).unsqueeze(-1).unsqueeze(-1)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, embed_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )
        self.embed = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )

    def forward(self, x1, x2, t):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = x + self.embed(t).unsqueeze(-1).unsqueeze(-1)
        return x


class SelfAttention(nn.Module):
    def __init__(self, channels, img_size):
        super().__init__()
        self.channels = channels
        self.img_size = img_size
        self.mha = nn.MultiheadAttention(channels, 4)
        self.ln = nn.LayerNorm(channels)
        self.ff_self = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):  # x: (n, c, h, w)
        x = x.view(-1, self.channels, self.img_size * self.img_size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attn_value, _ = self.mha(x_ln, x_ln, x_ln)
        attn_value = attn_value + x
        attn_value = self.ff_self(attn_value) + attn_value
        return attn_value.swapaxes(2, 1).view(
            -1, self.channels, self.img_size, self.img_size
        )


# Utils
def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(
        torch.cat(
            [
                torch.cat([i for i in images.cpu()], dim=-1),
            ],
            dim=-2,
        )
        .permute(1, 2, 0)
        .cpu()
    )
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    ndarr = (ndarr * 255).astype(np.uint8)
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(args.image_size + args.image_size // 4),
            torchvision.transforms.RandomResizedCrop(args.image_size, scale=(0.8, 1.0)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = torchvision.datasets.CIFAR100(transform=transforms, root=args.data_path)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


# Training
def train(args):
    setup_logging(args.run_name)
    device = args.device
    dataloader = get_data(args)
    model = UNet(img_size=args.image_size, device=device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("models", args.run_name))
    len_dl = len(dataloader)

    for epoch in range(args.epochs):
        logging.info(f"Epoch {epoch}")
        pbar = tqdm(dataloader, ncols=70)
        for i, (images, _) in enumerate(pbar):
            images = images.to(device)
            optimizer.zero_grad()

            t = diffusion.sample_timesteps(images.shape[0])
            x_t, noise = diffusion.forward_diffusion(images, t)
            noise_pred = model(x_t, t)
            loss = mse(noise_pred, noise)

            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss, epoch * len_dl + i)
            logging.info(f"Epoch {epoch} Batch {i} Loss {loss}")

            if (epoch * len_dl + i) % 1000 == 0:
                sampled_images = diffusion.backward_diffusion(model, 16)
                save_images(
                    sampled_images,
                    os.path.join("results", args.run_name, f"{epoch * len_dl + i}.png"),
                    nrow=4,
                    normalize=True,
                    value_range=(-1, 1),
                    padding=10,
                    pad_value=1,
                )
                torch.save(
                    model.state_dict(),
                    os.path.join("models", args.run_name, f"{epoch * len_dl + i}.pt"),
                )


def launch():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_CIFAR100_Orig_2e-5_large"
    args.epochs = 2000
    args.image_size = 64
    args.device = "cuda"
    args.data_path = "data"

    # if the outputs tend to be saturated to a single color (e.g., yellow) in a validation step, try to decrease the learning rate
    args.lr = 2e-5
    args.batch_size = 32

    train(args)


if __name__ == "__main__":
    launch()
