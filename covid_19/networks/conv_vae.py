# -*- coding: utf-8 -*-
"""
@created on: 4/3/21,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import tensor

flattened_size = 32 * 9 * 40  # 32 filters each of size 9*40 - reduced from a input size of 40*690 
mu_layer_nodes = 128


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
        self.fc_mu = nn.Linear(flattened_size, mu_layer_nodes)
        self.fc_var = nn.Linear(flattened_size, mu_layer_nodes)

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
        flattened = encoder_op5.view(encoder_op5.size(0), -1)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(flattened)
        log_var = self.fc_var(flattened)
        return mu, log_var


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()
        self.decoder_linear = nn.Linear(mu_layer_nodes, flattened_size)
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
        x = self.decoder_linear(x)
        x = x.view(x.size(0), 32, 9, 40)
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
        reconstructed_x = torch.sigmoid(self.decoder5_bn(self.decoder5(decoder_op4, output_size=out_size)))
        return reconstructed_x


class ConvVariationalAutoEncoder(nn.Module):

    def __init__(self):
        """
        You can add additional arguments as you need.
        In the constructor we instantiate modules and assign them as
        member variables.
        """

        super(ConvVariationalAutoEncoder, self).__init__()

        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        reconstructed_x = self.decoder(z, self.encoder.pool1_indices, self.encoder.pool2_indices,
                                       out_size=x.unsqueeze(1).size())
        return reconstructed_x, mu, log_var, z

    def sample(self, n):
        z = torch.randn((n, flattened_size))
        return self.decoder(z)
