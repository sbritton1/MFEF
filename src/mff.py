import torch
from torch import nn

# Multi-feature fusion (MFF) module
class MFF(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1x1 = torch.nn.Conv2d(3, 3, kernel_size=1, stride=1, padding='same')
        self.conv3x3 = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1, padding='same')
        self.conv5x5 = torch.nn.Conv2d(3, 3, kernel_size=5, stride=1, padding='same')

        self.gelu1 = torch.nn.GELU()
        self.gelu3 = torch.nn.GELU()
        self.gelu5 = torch.nn.GELU()

        # TODO: hoeveel groups?
        groups = 3
        self.shuffle = torch.nn.ChannelShuffle(groups)

        self.conv_final = torch.nn.Conv2d(9, 3, kernel_size=3, stride=1, padding='same')


    def forward(self, FB1, FB2, FB3):
        F1 = self.gelu1(self.conv1x1(FB1)) + FB1
        F2 = self.gelu3(self.conv3x3(FB2)) + FB2
        F3 = self.gelu5(self.conv5x5(FB3)) + FB3

        # print(F1.shape, F2.shape, F3.shape)
        Fk = torch.cat((F1,F2,F3))

        # Shuffle removed because derivative not implemented
        MFFk = self.conv_final(Fk)

        return MFFk