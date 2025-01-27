#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=pad,
        )

    def forward(self, x, h, c):
        # x: (B, input_channels, H, W)
        # h, c: (B, hidden_channels, H, W)
        combined = torch.cat([x, h], dim=1)  # (B, input+hidden, H, W)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """
    One-layer ConvLSTM -> final conv to predict NDVI channel.
    Input: (B, T, 4, H, W). Output: (B,1,H,W).
    """

    def __init__(self, in_channels=4, hidden_channels=32):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=3)
        self.out_conv = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.hidden_channels, H, W, device=x.device)
        c = torch.zeros(B, self.hidden_channels, H, W, device=x.device)

        for t in range(T):
            xt = x[:, t, :, :, :]  # (B,4,H,W)
            h, c = self.cell(xt, h, c)

        out = self.out_conv(h)  # (B,1,H,W)
        return out
