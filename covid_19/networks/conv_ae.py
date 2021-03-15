# -*- coding: utf-8 -*-
"""
@created on: 3/7/21,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

# -*- coding: utf-8 -*-
"""
@created on: 4/4/20,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import torch.nn as nn
import torch.nn.functional as F
from torch import tensor


class ConvEncoder(nn.Module):
    def __init__(self):
        super(ConvEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=[1, 2])
        self.conv1_bn = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=1, return_indices=True)
        self.dropout0 = nn.Dropout(p=0.4)
        #
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=[1, 2])
        self.conv4_bn = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=1, return_indices=True)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=32, kernel_size=3, stride=[1, 2])
        self.conv5_bn = nn.BatchNorm2d(32)
        self.pool1_indices = None
        self.pool2_indices = None

    def forward(self, x):
        x = x.unsqueeze(1)
        # print('x.shape ', x.shape)
        encoder_op1 = F.relu(self.conv1(x))
        # print('conv 1', encoder_op1.shape)
        encoder_op2 = F.relu(self.conv2(encoder_op1))
        # print('conv 2', encoder_op2.shape)
        encoder_op2_pool, self.pool1_indices = self.pool1(encoder_op2)
        # print('pool1', encoder_op2_pool.shape)
        encoder_op2_pool = self.dropout0(encoder_op2_pool)

        encoder_op3 = F.relu(self.conv3(encoder_op2_pool))
        # print('conv 3', encoder_op3.shape)
        encoder_op4 = F.relu(self.conv4(encoder_op3))
        # print('conv 4', encoder_op4.shape)
        encoder_op4_pool, self.pool2_indices = self.pool2(encoder_op4)
        # print('pool2 ', encoder_op4_pool.shape)

        encoder_op5 = F.relu(self.conv5(encoder_op4_pool))
        # print('after conv net 5 ', encoder_op5.shape)

        # Stack filter maps next to each other

        return encoder_op5


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.decoder1 = nn.ConvTranspose2d(in_channels=32, out_channels=128, kernel_size=3, stride=[1, 2])
        self.decoder1_bn = nn.BatchNorm2d(128)
        self.unpool1 = nn.MaxUnpool2d(4, stride=1)
        self.decoder2 = nn.ConvTranspose2d(in_channels=128, out_channels=256, kernel_size=3, stride=[1, 2])
        self.decoder2_bn = nn.BatchNorm2d(256)
        self.decoder3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=[2, 2])
        self.decoder3_bn = nn.BatchNorm2d(128)
        self.unpool2 = nn.MaxUnpool2d(4, stride=1)
        self.decoder4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1)
        self.decoder4_bn = nn.BatchNorm2d(64)
        self.decoder5 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=[1, 2])
        self.decoder5_bn = nn.BatchNorm2d(1)

    def forward(self, x, pool1_indices, pool2_indices, out_size):
        decoder_op1 = F.relu(self.decoder1_bn(self.decoder1(x)))  # , output_size=encoder_op4_pool.size()
        # print('decoder1', decoder_op1.size())
        decoder_op1_unpool1 = self.unpool1(decoder_op1, indices=pool2_indices)
        # print("decoder_op1_unpool1", decoder_op1_unpool1.size())
        decoder_op2 = F.relu(self.decoder2_bn(self.decoder2(decoder_op1_unpool1)))  # , output_size=encoder_op3.size()
        # print('decoder2', decoder_op2.size())
        decoder_op3 = F.relu(self.decoder3_bn(self.decoder3(decoder_op2)))
        # print('decoder3', decoder_op3.size())
        decoder_op3_unpool2 = self.unpool2(decoder_op3, indices=pool1_indices)
        # print("decoder_op3_unpool2", decoder_op3_unpool2.size())

        decoder_op4 = F.relu(self.decoder4_bn(self.decoder4(decoder_op3_unpool2)))
        # print('decoder4', decoder_op4.size())
        reconstructed_x = F.sigmoid(self.decoder5_bn(self.decoder5(decoder_op4, output_size=out_size)))
        return reconstructed_x


class ConvAutoEncoder(nn.Module):

    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """

        super(ConvAutoEncoder, self).__init__()

        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        latent_space = self.encoder(x)
        reconstructed_x = self.decoder(latent_space, self.encoder.pool1_indices, self.encoder.pool2_indices,
                                       out_size=x.unsqueeze(1).size())
        latent_space = latent_space.view(-1, latent_space.size()[1:].numel())
        return reconstructed_x, latent_space
