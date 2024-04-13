import torch
from torch import nn

# Residual enhancement module (REM)
class REM(nn.Module):
    #TODO make channels dynamic
    def __init__(self, input_channels):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')
        self.gelu1 = torch.nn.GELU()
        self.conv2 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')
        self.gelu2 = torch.nn.GELU()
        self.conv3 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')
        self.gelu3 = torch.nn.GELU()
        self.conv4 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')

        self.conv5 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')
        self.gelu4 = torch.nn.GELU()
        self.conv6 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')
        self.gelu5 = torch.nn.GELU()
        self.conv7 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')
        self.gelu6 = torch.nn.GELU()
        self.conv8 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding='same')


    def forward(self, input):
        x1 = self.conv1(input)
        x1 = self.gelu1(x1)
        x1 = self.conv2(x1)
        x1 = self.gelu2(x1)
        x1 = self.conv3(x1)
        x1 = self.gelu3(x1)
        x1 = self.conv4(x1)

        x_between = x1 + input

        x2 = self.conv5(x_between)
        x2 = self.gelu4(x2)
        x2 = self.conv6(x2)
        x2 = self.gelu5(x2)
        x2 = self.conv7(x2)
        x2 = self.gelu6(x2)
        x2 = self.conv8(x2)

        x_end = x2 + x_between

        return x_end