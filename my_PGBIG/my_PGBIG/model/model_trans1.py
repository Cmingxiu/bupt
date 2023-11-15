import torch
from torch import nn
import math
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import numpy as np
import random
from functools import partial
# from einops import rearrange
from timm.models.layers import DropPath

import matplotlib.pyplot as plt
import seaborn as sns


# Transformer implementation comes from https://github.com/zczcwh/PoseFormer
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, num_frame=10, num_joints=66, in_chans=3, embed_dim_ratio=8, depth=4,
                 num_heads=4, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2,  norm_layer=None):
        """    ##########hybrid_backbone=None, representation_size=None,
        Args:
            num_frame (int, tuple): input frame number
            num_joints (int, tuple): joints number
            in_chans (int): number of input channels, 2D joints have 2 channels: (x,y)
            embed_dim_ratio (int): embedding dimension ratio
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        embed_dim_ratio = num_heads
        embed_dim = embed_dim_ratio * num_joints   #### temporal embed_dim is num_joints * spatial embedding dim ratio
        # out_dim = num_joints * 3     #### output dimension is num_joints * 3

        ### spatial patch embedding
        # self.Spatial_patch_to_embedding = nn.Linear(in_chans, embed_dim_ratio)
        # self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, embed_dim_ratio))

        self.embedding = nn.Linear(num_joints, embed_dim)
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, num_frame, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        # self.Spatial_blocks = nn.ModuleList([
        #     TransformerBlock(
        #         dim=embed_dim_ratio, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #         drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
        #     for i in range(depth)])

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        # self.Spatial_norm = norm_layer(embed_dim_ratio)
        self.Temporal_norm = norm_layer(embed_dim)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_joints)
        )


    # def Spatial_forward_features(self, x):
    #     b, _, f, p = x.shape  ##### b is batch size, f is number of frames, p is number of joints
    #     x = rearrange(x, 'b c f p  -> (b f) p  c', )
    #
    #     x = self.Spatial_patch_to_embedding(x)
    #     x += self.Spatial_pos_embed
    #     x = self.pos_drop(x)
    #
    #     for blk in self.Spatial_blocks:
    #         x = blk(x)
    #
    #     x = self.Spatial_norm(x)
    #     x = rearrange(x, '(b f) w c -> b f (w c)', f=f)
    #     return x

    def forward_features(self, x):
        x = self.embedding(x)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.Temporal_norm(x)
        ##### x size [b, f, emb_dim], then take weighted mean on frame dimension, we only predict 3D pose of the center frame
        return x

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # b, _, p = x.shape
        ### now x is [batch_size, 2 channels, receptive frames, joint_num], following image data
        # x = self.Spatial_forward_features(x)
        x = self.forward_features(x)
        x = self.head(x)

        # x = x.view(b, 1, p, -1)

        return x.permute(0, 2, 1)


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
        self.act_f = nn.Sigmoid()

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
        
        # y11 = self.gc11(y10 + y1) if self.in_features == self.out_features else self.gc11(y10) + torch.matmul(self.conv1d(y1.permute(0, 2, 1)).permute(0, 2, 1), self.weight1)
        y11 = self.gc11(y10 + y1) if self.in_features == self.out_features else self.gc11(y10)
        b, n, f = y11.shape
        y11 = self.bn11(y11.view(b, -1)).view(b, n, f)
        y11 = self.act_f(y11)
        y11 = self.do(y11)
        
        # return y11 + x if self.in_features == self.out_features else y11 + torch.matmul(self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1), self.weight2)
        return y11 + x if self.in_features == self.out_features else y11

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class model(nn.Module):
    def __init__(self, opt):
        super(model, self).__init__()

        self.mask_ratio = opt.mask_ratio
        assert 0. < self.mask_ratio < 1., f'mask ratio must be kept between 0 and 1, got: {mask_ratio}'

        self.enlayers = opt.enlayers  # Transformer encoder个数
        self.num_heads = opt.num_heads  # 多头注意力头数
        self.input_feature = opt.input_feature  # 输入维度
        self.hidden_feature = opt.hidden_feature  # 隐层维度
        self.input_n = opt.input_n  # 输入帧数
        self.output_n = opt.output_n  # 输出帧数
        # self.train_batch = train_batch
        # self.num_masked = int(self.mask_ratio * self.input_feature)
        # self.num_unmasked = self.input_feature - self.num_masked
        self.node_n = opt.node_n  # 节点个数
        self.p_dropout = opt.drop_out  # drop_out率
        self.is_cuda = torch.cuda.is_available()  # CUDA
        
        # self.shuffle_indices = torch.rand(self.train_batch, self.input_feature).argsort()
        # if self.is_cuda:
        #     self.shuffle_indices = self.shuffle_indices.cuda()

        self.embedding0 = nn.Linear(self.input_n, self.hidden_feature)
        self.embedding1 = nn.Linear(self.node_n, self.hidden_feature)

        self.transformer = Transformer(num_frame=self.input_n, num_joints=self.node_n, depth=self.enlayers,
                                       num_heads=self.num_heads,
                                       drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2)

        self.SRB0 = GCN(self.hidden_feature, self.hidden_feature, p_dropout=self.p_dropout, node_n=self.hidden_feature)
        self.SRB1 = GCN(self.hidden_feature, self.hidden_feature, p_dropout=self.p_dropout, node_n=self.hidden_feature)
        self.SRB2 = GCN(2*self.hidden_feature, self.hidden_feature, p_dropout=self.p_dropout, node_n=self.node_n)
        
        self.linear0 = nn.Linear(self.hidden_feature, self.node_n)
        self.linear1 = nn.Linear(self.hidden_feature, self.node_n)

        self.final = nn.Linear(self.hidden_feature, self.output_n)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = x[:, :, :self.input_n]

        b, n, f = x.shape  # [8, 66, 10]
        x_um = x

        y_um = self.transformer(x_um) + x

        y_um = self.embedding0(y_um)  # 时间维度嵌入 [8, 66, 512]
        y_um = self.embedding1(y_um.permute(0, 2, 1)).permute(0, 2, 1)  # 空间维度嵌入 [8, 512, 512]
        
        y_t = self.SRB0(y_um.clone())  # 时间维度GCN
        y_s = self.SRB1(y_um.clone().permute(0, 2, 1)).permute(0, 2, 1)  # 空间维度GCN

        y_t = self.linear0(y_t.permute(0, 2, 1)).permute(0, 2, 1)  # [8, 66, 512]
        y_s = self.linear1(y_s.permute(0, 2, 1)).permute(0, 2, 1)  # [8, 66, 512]

        y = torch.cat([y_t, y_s], 2)  # [8, 66, 1024]

        y = self.SRB2(y)  # 融合GCN [8, 66, 512]
        
        y = self.final(y)
        y = y + x_um[:, :, -1, None]

        return y.permute(0, 2, 1)

