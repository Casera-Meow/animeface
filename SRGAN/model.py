
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels, 0.8),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels, 0.8)
        )

    def forward(self, x):
        conved = self.block(x)
        return conved + x

class UpSample(nn.Module):
    def __init__(self, channels=64):
        super(UpSample, self).__init__()

        self.up = nn.Sequential(
            nn.Conv2d(channels, channels*4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

    def forward(self, x):
        return self.up(x)

class Generator(nn.Module):
    def __init__(self, n_blocks=16):
        super(Generator, self).__init__()

        self.input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        blocks = []
        for _ in range(n_blocks):
            blocks.append(ResBlock(channels=64))
        self.res_blocks = nn.Sequential(
            *blocks
        )

        self.inter = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8)
        )

        self.up = nn.Sequential(
            UpSample(channels=64),
            UpSample(channels=64)
        )

        channel = 256 // 4

        self.outer = nn.Sequential(
            nn.Conv2d(channel, 3, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    def forward(self, img):

        img0 = self.input(img)
        img = self.res_blocks(img0)
        img = self.inter(img)
        img = img + img0
        img = self.up(img)
        img = self.outer(img)

        return img

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2)
        )

        self.blocks = nn.Sequential(
            ConvBlock(64, 64, 2),
            ConvBlock(64, 128, 1),
            ConvBlock(128, 128, 2),
            ConvBlock(128, 256, 1),
            ConvBlock(256, 256, 2),
            ConvBlock(256, 512, 1),
            ConvBlock(512, 512, 2)
        )

        self.outer = nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, img):
        x = self.input(img)
        x = self.blocks(x)
        x = self.outer(x)

        return x

if __name__ == "__main__":
    G = Generator()

    img = torch.randn(1, 3, 256//4, 256//4)
    print(img.size())

    output = G(img)

    print(output.size())

    D = Discriminator()

    output = D(output)

    print(output.size())

    padded = torch.nn.functional.pad(img, pad=(96, 96, 96, 96), mode='constant', value=-1.)

    print(padded.size())
