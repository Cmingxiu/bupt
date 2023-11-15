from torch.nn import Module
from torch import nn
import torch
# import model.transformer_base
import math
from model import model_trans as BaseBlock
import utils.util as util
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from utils.h36m_opt import Options



class MultiStageModel(Module):
    def __init__(self, opt):
        super(MultiStageModel, self).__init__()

        self.opt = opt

        self.baseblock1 = BaseBlock.model(opt)

        self.baseblock2 = BaseBlock.model(opt)

        self.baseblock3 = BaseBlock.model(opt)

        self.baseblock4 = BaseBlock.model(opt)

    def forward(self, src):

        # stage1
        initial_1 = self.baseblock1(src)

        # stage2
        initial_2 = self.baseblock2(initial_1)

        # stage3
        initial_3 = self.baseblock3(initial_2)

        # stage4
        initial_4 = self.baseblock4(initial_3)

        return initial_4, initial_3, initial_2, initial_1  # [32, 10, 66 ]


if __name__ == '__main__':
    option = Options().parse()
    model = MultiStageModel(opt=option).cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    src = torch.FloatTensor(torch.randn((32, 35, 66))).cuda()
    output, att_map, zero = model(src)
