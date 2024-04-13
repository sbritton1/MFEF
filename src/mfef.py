import torch
from torch import nn

from .rem import REM
from .mff import MFF
from .dsdc import DSDC
from .pcam import PCAM

# Final network
class MFEF(nn.Module):

    def __init__(self):
        super().__init__()

        self.rem1_1 = REM(3)
        self.rem1_2 = REM(3)
        self.rem1_3 = REM(3)

        self.mff1 = MFF()
        self.mff2 = MFF()
        self.mff3 = MFF()

        self.rem2_1 = REM(9)
        self.rem2_2 = REM(6)
        self.rem2_3 = REM(3)

        self.dim_reduce1 = torch.nn.Conv2d(6, 3, kernel_size=3, stride=1, padding='same')
        self.dim_reduce2 = torch.nn.Conv2d(9, 3, kernel_size=3, stride=1, padding='same')

        self.dsdc = DSDC()

        self.pcam1 = PCAM(9)
        self.pcam2 = PCAM(9)
        self.pcam3 = PCAM(9)

        self.upsampling = torch.nn.Upsample(scale_factor=2, mode='nearest')

        self.rem3_1 = REM(27)
        self.rem3_2 = REM(18)
        self.rem3_3 = REM(9)

        self.conv = torch.nn.Conv2d(27, 3, kernel_size=3, stride=1, padding='same')

    def forward(self, input_orig, input_wb, input_clahe):


        rem1_1_res = self.rem1_1(input_orig)
        rem1_2_res = self.rem1_2(input_wb)
        rem1_3_res = self.rem1_3(input_clahe)

        mff1_res = self.mff1(rem1_1_res, rem1_2_res, rem1_3_res)
        mff2_res = self.mff2(rem1_1_res, rem1_2_res, rem1_3_res)
        mff3_res = self.mff3(rem1_1_res, rem1_2_res, rem1_3_res)

        rem2_3_res = self.rem2_3(mff3_res)
        # print(torch.cat((mff2_res, rem2_3_res), dim=0).shape)
        rem2_2_res = self.rem2_2(torch.cat((mff2_res, rem2_3_res), dim=0))
        rem2_1_res = self.rem2_1(torch.cat((mff3_res, rem2_2_res), dim=0))

        dsdc_1_in = input_orig + self.dim_reduce2(rem2_1_res)
        dsdc_2_in = input_wb + self.dim_reduce1(rem2_2_res)
        dsdc_3_in = input_clahe + rem2_3_res

        dsdc_1_out, dsdc_2_out, dsdc_3_out = self.dsdc(dsdc_1_in, dsdc_2_in, dsdc_3_in)
        # print(dsdc_1_in.shape, dsdc_1_out.shape)
        # print(dsdc_2_in.shape, dsdc_2_out.shape)
        # print(dsdc_3_in.shape, dsdc_3_out.shape)


        pcam1_res = self.pcam1(dsdc_1_out)
        pcam2_res = self.pcam2(dsdc_2_out)
        pcam3_res = self.pcam3(dsdc_3_out)

        # print("---- MFEF ----")

        full_size = pcam1_res.shape
        half_size = pcam2_res.shape
        # print("pcam1", pcam1_res.shape)
        # print("pcam2", pcam2_res.shape)
        # print("pcam3", pcam3_res.shape)


        rem3_3_res = self.rem3_3(pcam3_res)
        rem3_3_res = rem3_3_res
        # print("rem3_3", rem3_3_res.shape)

        rem3_3_res_up = self.upsampling(rem3_3_res.unsqueeze(0)).squeeze()
        # print("rem3_3_up", rem3_3_res_up.shape)
        # if size is not correct, crop off the added padding
        if rem3_3_res_up.shape != full_size:
            rem3_3_res_up = rem3_3_res_up[:, :half_size[1], :half_size[2]]
            # print("rem3_3_up", rem3_3_res_up.shape)

        rem3_2_res = self.rem3_2(torch.cat((pcam2_res, rem3_3_res_up)))
        # print("rem3_2", rem3_2_res.shape)

        rem3_2_res_up = self.upsampling(rem3_2_res.unsqueeze(0)).squeeze()
        # print("rem3_2_up", rem3_2_res_up.shape)
        if rem3_2_res_up.shape != full_size:
            rem3_2_res_up = rem3_2_res_up[:, :full_size[1], :full_size[2]]
            # print("rem3_3_up", rem3_3_res_up.shape)

        rem3_1_res = self.rem3_1(torch.cat((pcam1_res, rem3_2_res_up)))
        # print("rem3_1", rem3_1_res.shape)
        #
        out = self.conv(rem3_1_res)
        # print("out", out.shape)
        return out