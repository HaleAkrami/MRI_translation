""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from unet_parts import *


import torch.nn.functional as F

from unet_parts import *


class UNet(nn.Module):
    def __init__(self, drop_prob, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(drop_prob, n_channels, 64)
        self.down1 = Down(drop_prob, 64, 128)
        self.down2 = Down(drop_prob, 128, 256)
        self.down3 = Down(drop_prob, 256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(drop_prob, 512, 1024 // factor)
        self.up1 = Up(drop_prob, 1024, 512 // factor, bilinear)
        self.up2 = Up(drop_prob, 512, 256 // factor, bilinear)
        self.up3 = Up(drop_prob, 256, 128 // factor, bilinear)
        self.up4 = Up(drop_prob, 128, 64, bilinear)
        self.outc1 = OutConv(64, n_classes)
        self.outc2 = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc1(x)
        varlogits = self.outc2(x)
        return logits, varlogits

    def weight_reset(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0, std=.02)
                #nn.init.xavier_uniform(m.weight)
                #nn.init.xavier_uniform(m.bias)