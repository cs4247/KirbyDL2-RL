import torch.nn as nn
import copy
import torch
import numpy

# based on Mario pytorch tutorial by yfeng997, implemented mostly the same was as the last kirby project in Pyboy-RL/model.py
# This was changed only to support the new input frame of the full screen of shape (144,160,3) with finer image features

class DDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim  # Unpacking the input dimensions: channels, height, width

        # Building the online network with adjusted convolutional layers for larger input
        self.online = nn.Sequential(
            # First convolutional layer with more channels and larger kernel for bigger image size
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),  # ReLU activation function to introduce non-linearity

            # Second convolutional layer with increased number of channels for more complex feature extraction
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),  # ReLU activation

            # Third convolutional layer with same number of channels but smaller kernel for finer feature extraction
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # ReLU activation

            nn.Flatten(),  # Flattening the output of conv layers to feed into linear layers

            # First linear layer; input features are dynamically calculated based on conv layer output
            nn.Linear(self._conv_output(input_dim), 512),
            nn.ReLU(), 
            nn.Linear(512, output_dim)
        )

        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):

        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)

    def _conv_output(self, shape):
        # Method to calculate the output size of the conv layers for the linear layer's input
        o = self.online[:6](torch.zeros(1, *shape))
        return int(numpy.prod(o.size()))