import torch
from torch import nn

class DSDC(nn.Module):

  def __init__(self):
    super().__init__()

  def forward(self, F1, F2, F3):
    # print("---- DSDC ----")
    # print("F1 shape", F1.shape)
    # print("F2 shape", F2.shape)
    # print("F3 shape", F3.shape)
    F_largest = torch.cat((F1, F2, F3))
    # Calculate padding
    # Padding format is [left, right, top, bottom] for the last two dimensions
    padding_large = [0, 1 if F_largest.size(2) % 2 != 0 else 0,  # Padding for width (right)
                    0, 1 if F_largest.size(1) % 2 != 0 else 0]  # Padding for height (bottom)

    F_largest_padded = torch.nn.functional.pad(F_largest, pad=padding_large, mode='constant', value=0)

    F_middle = F_largest_padded[::,::2,::2]

    padding_medium =    [0, 1 if F_middle.size(2) % 2 != 0 else 0,  # Padding for width (right)
                        0, 1 if F_middle.size(1) % 2 != 0 else 0]  # Padding for height (bottom)

    F_middle_padded = torch.nn.functional.pad(F_middle, pad=padding_medium, mode='constant', value=0)

    F_smallest = F_middle_padded[::,::2,::2]


    # print("F_largest shape", F_largest.shape)
    # print("F_middle shape", F_middle.shape)
    # print("F_smallest shape", F_smallest.shape)

    return F_largest, F_middle, F_smallest