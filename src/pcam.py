import torch
from torch import nn

# Pixel-weighted channel attention module (PCAM)
class PCAM(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.conv3_1 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.gelu1 = torch.nn.GELU()
        self.conv1 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.conv3_2 = torch.nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.sigmoid1 = torch.nn.Sigmoid()
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        # fc_size = 125*166
        fc_size = 100
        self.fc1 = torch.nn.Linear(input_channels, input_channels)
        self.gelu2 = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(input_channels, input_channels)
        self.sigmoid2 = torch.nn.Sigmoid()


    def forward(self, F_in):
      # print("---- PCAM ----")
      # print(f"F_in Shape: {F_in.shape}")
      F_n = self.gelu1(self.conv3_1(F_in))
#       print(f"F_n Shape: {F_n.shape}")
      F = torch.mul(self.sigmoid1(self.conv1(F_n)), self.conv3_2(F_n))
#       print(f"F Shape: {F.shape}")
      O = self.gap(F).view(9)
#       print(f"O Shape: {O.shape}")
      f1 = self.gelu2(self.fc1(O))
#       print(f"f1 Shape: {f1.shape}")
      f2 = self.sigmoid2(self.fc2(f1))
#       print(f"f2 Shape: {f2.shape}")
      F_out = F + F_in + F_n * f2[:, None, None]
      # print(f"F_out Shape: {F_out.shape}")
      return F_out