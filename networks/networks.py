import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.utils import AverageMeter, get_scheduler, get_gan_loss, psnr, get_nonlinearity
from networks.SE import *
import pdb


# Dense Block
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, norm='None'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm3d(growthRate)


    def forward(self, x):
        out = self.conv(x)
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out


# Residual dense block
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)

        out = self.conv_1x1(out)

        out = out + x # Residual
        return out


'''
spatial-channel Squeeze and Excite Residual Dense UNet
'''
class scSERDUNet(nn.Module):
    def __init__(self, n_channels=1, n_filters=64, n_denselayer=6, growth_rate=32, norm='None'):
        super(scSERDUNet, self).__init__()

        self.conv1 = nn.Conv3d(n_channels, n_filters, kernel_size=3, padding=1, bias=True)

        # encode
        self.RDB1 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE1 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB2 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE2 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.RDB3 = RDB(n_filters*1, n_denselayer, growth_rate, norm)
        self.SE3 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        # decode
        self.up3 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.RDB4 = RDB(n_filters*1, n_denselayer, growth_rate, norm)

        self.SE4 = ChannelSpatialSELayer3D(n_filters*1, norm='None')

        self.up4 = nn.ConvTranspose3d(n_filters*1, n_filters*1, kernel_size=2, stride=2)
        self.RDB5 = RDB(n_filters*1, n_denselayer, growth_rate, norm)

        self.conv2 = nn.Conv3d(n_filters, 1, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)

        # encode
        RDB1 = self.RDB1(x)
        SE1 = self.SE1(RDB1)
        x = F.avg_pool3d(SE1, 2)

        RDB2 = self.RDB2(x)
        SE2 = self.SE2(RDB2)
        x = F.avg_pool3d(SE2, 2)

        RDB3 = self.RDB3(x)
        SE3 = self.SE3(RDB3)

        # decode
        up3 = self.up3(SE3)

        RDB4 = self.RDB4(up3 + SE2)
        SE4 = self.SE4(RDB4)

        up4 = self.up4(SE4)

        RDB5 = self.RDB5(up4 + SE1)

        output = self.conv2(RDB5)

        return output


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)


if __name__ == '__main__':
    pass

