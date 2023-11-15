import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import random
# from model import Transformer_block


class PositionalEncoding(nn.Module):
    """adapted"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


def whole_joint_masking(intoken, num_masked, patches, is_cuda):
    b, n, f = patches.size()

    # Shuffle:生成对应 patch 的随机索引
    # torch.rand() 服从均匀分布(normal distribution)
    # torch.rand() 只是生成随机数，argsort() 是为了获得成索引
    # (b, n_patches)
    shuffle_indices = torch.rand(b, intoken).argsort()
    if is_cuda:
        shuffle_indices = shuffle_indices.cuda()
    # mask 和 unmasked patches 对应的索引
    mask_ind, unmask_ind = shuffle_indices[:, :num_masked], shuffle_indices[:, num_masked:]

    # 对应 batch 维度的索引：(b,1)
    batch_ind = torch.arange(b).unsqueeze(-1)
    if is_cuda:
        batch_ind = batch_ind.cuda()
    
    # 利用先前生成的索引对 patches 进行采样，分为 mask 和 unmasked 两组
    mask_patches, unmask_patches = patches[batch_ind, :, mask_ind], patches[batch_ind, :, unmask_ind]
    return mask_patches.permute(0, 2, 1), unmask_patches.permute(0, 2, 1), batch_ind, shuffle_indices


class TransformerEncoderModel(nn.Module):
    def __init__(self, intoken, hidden, enlayers, is_cuda=False, dropout=0.1):
        super(TransformerEncoderModel, self).__init__()

        nhead = hidden // 128

        self.intoken = intoken
        self.is_cuda = is_cuda

        encoder_layers = nn.TransformerEncoderLayer(self.intoken, nhead, hidden, dropout)
        self.transformerencoder = nn.TransformerEncoder(encoder_layers, enlayers)

        self.src_mask = None

    def forward(self, src):
        tgt = self.transformerencoder(src, src_key_padding_mask=self.src_mask)
        return tgt


# GCN implementation comes from https://github.com/wei-mao-2019/LearnTrajDep
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        # self.att = Parameter(torch.FloatTensor(node_n, node_n))
        self.att = Parameter(torch.FloatTensor(0.01 + 0.99 * np.eye(node_n)[np.newaxis, ...]))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight.data.uniform_(-stdv, stdv)
        # self.att.data.uniform_(-stdv, stdv)
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            # self.bias.data.uniform_(-stdv, stdv)
            self.bias.data.zero_()

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, in_features, out_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.gc3 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn3 = nn.BatchNorm1d(node_n * in_features)

        self.gc4 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn4 = nn.BatchNorm1d(node_n * in_features)

        self.gc5 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn5 = nn.BatchNorm1d(node_n * in_features)

        self.gc6 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn6 = nn.BatchNorm1d(node_n * in_features)

        self.gc7 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn7 = nn.BatchNorm1d(node_n * in_features)

        self.gc8 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn8 = nn.BatchNorm1d(node_n * in_features)

        self.gc9 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn9 = nn.BatchNorm1d(node_n * in_features)

        self.gc10 = GraphConvolution(in_features, out_features, node_n=node_n, bias=bias)
        self.bn10 = nn.BatchNorm1d(node_n * out_features)
        
        self.gc11 = GraphConvolution(out_features, out_features, node_n=node_n, bias=bias)
        self.bn11 = nn.BatchNorm1d(node_n * out_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

        self.conv1d = nn.Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1)

    def forward(self, x):
        y1 = self.gc1(x)
        b, n, f = y1.shape
        y1 = self.bn1(y1.view(b, -1)).view(b, n, f)
        y1 = self.act_f(y1)
        y1 = self.do(y1)

        y2 = self.gc2(y1)
        b, n, f = y2.shape
        y2 = self.bn2(y2.view(b, -1)).view(b, n, f)
        y2 = self.act_f(y2)
        y2 = self.do(y2)

        y3 = self.gc3(y2)
        b, n, f = y3.shape
        y3 = self.bn3(y3.view(b, -1)).view(b, n, f)
        y3 = self.act_f(y3)
        y3 = self.do(y3)

        y4 = self.gc4(y3)
        b, n, f = y4.shape
        y4 = self.bn4(y4.view(b, -1)).view(b, n, f)
        y4 = self.act_f(y4)
        y4 = self.do(y4)

        y5 = self.gc5(y4)
        b, n, f = y5.shape
        y5 = self.bn5(y5.view(b, -1)).view(b, n, f)
        y5 = self.act_f(y5)
        y5 = self.do(y5)

        y6 = self.gc6(y5)
        b, n, f = y6.shape
        y6 = self.bn6(y6.view(b, -1)).view(b, n, f)
        y6 = self.act_f(y6)
        y6 = self.do(y6)

        y7 = self.gc7(y6 + y5)
        b, n, f = y7.shape
        y7 = self.bn7(y7.view(b, -1)).view(b, n, f)
        y7 = self.act_f(y7)
        y7 = self.do(y7)

        y8 = self.gc8(y7 + y4)
        b, n, f = y8.shape
        y8 = self.bn8(y8.view(b, -1)).view(b, n, f)
        y8 = self.act_f(y8)
        y8 = self.do(y8)

        y9 = self.gc9(y8 + y3)
        b, n, f = y9.shape
        y9 = self.bn9(y9.view(b, -1)).view(b, n, f)
        y9 = self.act_f(y9)
        y9 = self.do(y9)

        y10 = self.gc10(y9 + y2)
        b, n, f = y10.shape
        y10 = self.bn10(y10.view(b, -1)).view(b, n, f)
        y10 = self.act_f(y10)
        y10 = self.do(y10)

        y11 = self.gc11(y10 + y1) if self.in_features == self.out_features else self.gc11(y10) + self.conv1d(y1.permute(0, 2, 1)).permute(0, 2, 1)
        b, n, f = y11.shape
        y11 = self.bn11(y11.view(b, -1)).view(b, n, f)
        y11 = self.act_f(y11)
        y11 = self.do(y11)
        
        return y11 + x if self.in_features == self.out_features else y11 + self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class model(nn.Module):
    def __init__(self, opt):
        super(model, self).__init__()

        self.mask_ratio = opt.mask_ratio
        assert 0. < self.mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'

        self.pre_train = opt.pre_train

        self.enlayers = opt.enlayers
        self.input_feature = opt.input_feature
        self.hidden_feature = opt.hidden_feature
        self.input_n = opt.input_n
        self.output_n = opt.output_n
        # self.train_batch = train_batch
        self.num_masked = int(self.mask_ratio * self.input_feature)
        self.num_unmasked = self.input_feature - self.num_masked
        self.node_n = opt.node_n
        self.p_dropout = opt.drop_out
        self.is_cuda = torch.cuda.is_available()

        
        # self.shuffle_indices = torch.rand(self.train_batch, self.input_feature).argsort()
        # if self.is_cuda:
        #     self.shuffle_indices = self.shuffle_indices.cuda()

        self.embedding0 = nn.Linear(self.input_n, self.hidden_feature)
        self.embedding1 = nn.Linear(self.node_n, self.hidden_feature)

        self.pos_encoder = PositionalEncoding(self.hidden_feature)
        self.transformer_encoder = TransformerEncoderModel(intoken=self.hidden_feature, hidden=self.hidden_feature,
                                                           enlayers=self.enlayers, is_cuda=self.is_cuda, dropout=self.p_dropout)

        self.SRB0 = GCN(self.hidden_feature, self.hidden_feature, p_dropout=self.p_dropout, node_n=self.hidden_feature)
        self.SRB1 = GCN(self.hidden_feature, self.hidden_feature, p_dropout=self.p_dropout, node_n=self.hidden_feature)
        self.SRB2 = GCN(2*self.hidden_feature, self.hidden_feature, p_dropout=self.p_dropout, node_n=self.node_n)
        
        self.linear0 = nn.Linear(self.hidden_feature, self.node_n)
        self.linear1 = nn.Linear(self.hidden_feature, self.node_n)

        self.final0 = nn.Linear(self.hidden_feature, self.output_n)
        self.final1 = nn.Linear(self.hidden_feature, self.output_n)

        self.mask_embed = nn.Parameter(35 * torch.randn(self.input_n))
        self.mask_embed1 = nn.Parameter(35 * torch.randn(self.input_n))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x[:, :, :self.input_n]

        b, n, f = x.shape  # [8, 66, 10]

        if self.pre_train:
            x_m = x.permute(0, 2, 1)
            mask_patches, unmask_patches, batch_ind, shuffle_indices = \
                whole_joint_masking(self.input_feature, self.num_masked, x_m, self.is_cuda)
            # [16, 10, 16], [16, 10, 50], [16, 1], [16, 66]

            mask_tokens = self.mask_embed[None, None, :].repeat(b, self.num_masked, 1).permute(0, 2, 1)  # [16, 10, 16]
            x_m = torch.cat([mask_tokens, unmask_patches], dim=2)
            tmp = torch.empty_like(x_m).permute(0, 2, 1)
            tmp[batch_ind, shuffle_indices] = x_m.permute(0, 2, 1)
            x_m = tmp

            y_m = self.embedding0(x_m)

            y_m = self.pos_encoder(y_m)
            y_m = self.transformer_encoder(y_m)

            y_m = self.SRB(y_m)

            y_m = self.final0(y_m)
            y_m = y_m + x_m[:, :, -1, None]

            return y_m[:, :, :self.output_n]

        else:
            a = random.random()
            if a > 1:
              x_um = x.permute(0, 2, 1)
              mask_patches, unmask_patches, batch_ind, shuffle_indices = \
                  whole_joint_masking(self.input_feature, self.num_masked, x_um, self.is_cuda)
              mask_tokens = self.mask_embed1[None, None, :].repeat(b, self.num_masked, 1).permute(0, 2, 1)
              x_um = torch.cat([mask_tokens, unmask_patches], dim=2)
              tmp = torch.empty_like(x_um).permute(0, 2, 1)
              tmp[batch_ind, shuffle_indices] = x_um.permute(0, 2, 1)
              x_um = tmp
            else:
              x_um = x

            y_um = self.embedding0(x_um)
            y_um = self.embedding1(y_um.permute(0, 2, 1)).permute(0, 2, 1)

            y_um = self.pos_encoder(y_um)
            res = y_um.clone()
            y_um = self.transformer_encoder(y_um)
            y_um = y_um + res

            y_t = self.SRB0(y_um)
            y_s = self.SRB1(y_um.permute(0, 2, 1)).permute(0, 2, 1)

            y_t = self.linear0(y_t.permute(0, 2, 1)).permute(0, 2, 1)
            y_s = self.linear1(y_s.permute(0, 2, 1)).permute(0, 2, 1)

            y = torch.cat([y_t, y_s], 2)
            
            y = self.SRB2(y)

            y = self.final1(y)
            y = y + x_um[:, :, -1, None]

            return y.permute(0, 2, 1)

