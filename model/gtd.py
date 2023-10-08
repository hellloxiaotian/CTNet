'''
    Group + Dynamic + transformer  + Denosing
'''

import sys

sys.path.append('../')

import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy


def make_model(args):
    return GDTD(args), 1


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class ResBlock(nn.Module):
    def __init__(
                self, conv, n_feats, kernel_size,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class S_ResBlock(nn.Module):
    def __init__(
                self, conv, n_feats, kernel_size,
                bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(S_ResBlock, self).__init__()

        assert len(conv) == 2

        m = []

        for i in range(2):
            m.append(conv[i](n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class GDTD(nn.Module):

    def __init__(self, args, conv=default_conv):
        super(GDTD, self).__init__()

        self.scale_idx = 0

        self.args = args

        n_feats = args.n_feats
        kernel_size = 3
        act = nn.ReLU(True)

        self.head1 = conv(args.n_colors, n_feats, kernel_size)

        self.head1_1 = ResBlock(conv, n_feats, kernel_size, act=act)

        self.head1_3 = VisionEncoder(img_dim=args.patch_size,
                                     patch_dim=args.patch_dim,
                                     num_channels=n_feats,
                                     embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                     num_heads=args.num_heads,
                                     num_layers=1,
                                     hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                     dropout_rate=args.dropout_rate,
                                     mlp=True,
                                     pos_every=args.pos_every,
                                     no_pos=args.no_pos,
                                     no_norm=args.no_norm,
                                     no_residual=args.no_residual
                                     )

        self.body1_1 = nn.Sequential(
            ResBlock(conv, n_feats, kernel_size, act=act),
            conv(n_feats, n_feats, kernel_size),   # conv3
            act
        )

        self.body1_2 = nn.Sequential(
            VisionEncoder(img_dim=args.patch_size,
                          patch_dim=args.patch_dim,
                          num_channels=n_feats,
                          embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                          num_heads=args.num_heads,
                          num_layers=1,
                          hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                          dropout_rate=args.dropout_rate,
                          mlp=True,
                          pos_every=args.pos_every,
                          no_pos=args.no_pos,
                          no_norm=args.no_norm,
                          no_residual=args.no_residual
                          ),
            ResBlock(conv, n_feats, kernel_size, act=act)
        )

        self.body2_1 = nn.Sequential(
            ResBlock(conv, n_feats, kernel_size, act=act),
            VisionEncoder(img_dim=args.patch_size,
                          patch_dim=args.patch_dim,
                          num_channels=n_feats,
                          embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                          num_heads=args.num_heads,
                          num_layers=1,
                          hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                          dropout_rate=args.dropout_rate,
                          mlp=True,
                          pos_every=args.pos_every,
                          no_pos=args.no_pos,
                          no_norm=args.no_norm,
                          no_residual=args.no_residual
                          )
        )

        self.fusion2_1 = nn.Sequential(
            # DConv
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
            conv(n_feats * 2, n_feats, 1)  # conv1
        )

        self.body2_2 = nn.Sequential(
            ResBlock(conv, n_feats, kernel_size, act=act),
            VisionEncoder(img_dim=args.patch_size,
                          patch_dim=args.patch_dim,
                          num_channels=n_feats,
                          embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                          num_heads=args.num_heads,
                          num_layers=1,
                          hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                          dropout_rate=args.dropout_rate,
                          mlp=True,
                          pos_every=args.pos_every,
                          no_pos=args.no_pos,
                          no_norm=args.no_norm,
                          no_residual=args.no_residual
                          )
        )

        self.fusion2_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
            conv(n_feats * 2, n_feats, 1),  # conv1
        )

        self.body2_3 = VisionEncoder(img_dim=args.patch_size,
                                     patch_dim=args.patch_dim,
                                     num_channels=n_feats,
                                     embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                     num_heads=args.num_heads,
                                     num_layers=1,
                                     hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                     dropout_rate=args.dropout_rate,
                                     mlp=True,
                                     pos_every=args.pos_every,
                                     no_pos=args.no_pos,
                                     no_norm=args.no_norm,
                                     no_residual=args.no_residual
                                     )

        self.body3_1 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),   # conv1
            act,
            VisionEncoder(img_dim=args.patch_size,
                          patch_dim=args.patch_dim,
                          num_channels=n_feats,
                          embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                          num_heads=args.num_heads,
                          num_layers=2,
                          hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                          dropout_rate=args.dropout_rate,
                          mlp=True,
                          pos_every=args.pos_every,
                          no_pos=args.no_pos,
                          no_norm=args.no_norm,
                          no_residual=args.no_residual
                          )
        )

        self.fusion3_1 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
            conv(n_feats * 2, n_feats, 1),  # conv1
        )

        self.body3_2 = nn.Sequential(
            conv(n_feats, n_feats, kernel_size),   # conv1
            act,
            VisionEncoder(img_dim=args.patch_size,
                          patch_dim=args.patch_dim,
                          num_channels=n_feats,
                          embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                          num_heads=args.num_heads,
                          num_layers=2,
                          hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                          dropout_rate=args.dropout_rate,
                          mlp=True,
                          pos_every=args.pos_every,
                          no_pos=args.no_pos,
                          no_norm=args.no_norm,
                          no_residual=args.no_residual
                          )
        )

        self.fusion3_2 = nn.Sequential(
            nn.Conv2d(n_feats*2, n_feats*2, kernel_size=5, padding=2, groups=n_feats*2),
            conv(n_feats * 2, n_feats, 1),  # conv1
        )

        self.body3_3 = VisionEncoder(img_dim=args.patch_size,
                                     patch_dim=args.patch_dim,
                                     num_channels=n_feats,
                                     embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                     num_heads=args.num_heads,
                                     num_layers=1,
                                     hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                     dropout_rate=args.dropout_rate,
                                     mlp=args.no_mlp,
                                     pos_every=args.pos_every,
                                     no_pos=True,
                                     no_norm=args.no_norm,
                                     no_residual=args.no_residual
                                     )

        self.fusion3_3 = nn.Sequential(
            nn.Conv2d(n_feats*3, n_feats*3, kernel_size=5, padding=2, groups=n_feats*3),
            conv(n_feats * 3, n_feats, 1),  # conv1
        )

        self.body3_4 = VisionEncoder(img_dim=args.patch_size,
                                     patch_dim=args.patch_dim,
                                     num_channels=n_feats,
                                     embedding_dim=n_feats * args.patch_dim * args.patch_dim,
                                     num_heads=args.num_heads,
                                     num_layers=1,
                                     hidden_dim=n_feats * args.patch_dim * args.patch_dim * 4,
                                     dropout_rate=args.dropout_rate,
                                     mlp=args.no_mlp,
                                     pos_every=args.pos_every,
                                     no_pos=True,
                                     no_norm=args.no_norm,
                                     no_residual=args.no_residual
                                     )

        self.tail = conv(n_feats, args.n_colors, kernel_size)

    def forward(self, x):
        y = x

        x = self.head1(x)
        x = self.head1_1(x)
        x = self.head1_3(x)

        group1 = self.body1_1(x)
        group2 = self.body2_1(x)
        group3 = self.body3_1(x)

        group2 = self.body2_2(self.fusion2_1(torch.cat((group1, group2), 1)))
        group3 = self.body3_2(self.fusion3_1(torch.cat((group2, group3), 1)))
        group1 = self.body1_2(group1)

        group2 = self.body2_3(self.fusion2_2(torch.cat((group1, group2), 1)))
        group3 = self.body3_3(self.fusion3_2(torch.cat((group2, group3), 1)))

        group3 = self.body3_4(self.fusion3_3(torch.cat((group1, group2, group3), 1)))

        x = group3

        out = self.tail(x)

        return y - out



class VisionEncoder(nn.Module):
    def __init__(
            self,
            img_dim,
            patch_dim,
            num_channels,
            embedding_dim,
            num_heads,
            num_layers,
            hidden_dim,
            dropout_rate=0,
            no_norm=False,
            mlp=False,
            pos_every=False,
            no_pos=False,
            no_residual=False
    ):
        super(VisionEncoder, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels

        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)  
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels  

        self.out_dim = patch_dim * patch_dim * num_channels

        self.no_pos = no_pos
        self.no_residual = no_residual

        if self.mlp == False: 
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )

        encoder_layer = TransformerEncoderLayer(patch_dim, embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)

        self.encoder = TransformerEncoder(encoder_layer, num_layers, self.no_residual)


        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )

        self.dropout_layer1 = nn.Dropout(dropout_rate)

        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=1 / m.weight.size(1))

    def forward(self, x, con=False, mask=None):

        x = torch.nn.functional.unfold(x, self.patch_dim, stride=self.patch_dim).transpose(1, 2).transpose(0, 1).contiguous()  # shape == (time, B, d_model)

        if self.mlp == False:
            x = self.dropout_layer1(self.linear_encoding(x)) + x
            # query_embed = self.query_embed.weight[query_idx].view(-1, 1, self.embedding_dim).repeat(1, x.size(1), 1)
        else:
            pass
            # query_embed = None

        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0, 1)

        if self.pos_every:
            x = self.encoder(x, pos=pos, mask=mask)
            # x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x, mask)
            # x = self.decoder(x, x, query_pos=query_embed)
        else:
            x = self.encoder(x + pos, mask)
            # x = self.decoder(x, x, query_pos=query_embed)

        if self.mlp == False:
            x = self.mlp_head(x) + x

        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)

        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                         stride=self.patch_dim)
            return x, con_x

        x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), int(self.img_dim), self.patch_dim,
                                     stride=self.patch_dim)

        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]  # self.position_ids???????

        position_embeddings = self.pe(position_ids)
        return position_embeddings


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, no_residual=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.no_residual = no_residual

    def forward(self, src, pos=None, mask=None):
        output = src

        if self.no_residual or len(self.layers) < 4:
            for layer in self.layers:
                output = layer(output, pos=pos, mask=mask)
        else:  # encoder use residual struct
            layers = iter(self.layers)

            output1 = next(layers)(output, pos=pos, mask=mask)
            output2 = next(layers)(output1, pos=pos, mask=mask)
            output3 = next(layers)(output2, pos=pos, mask=mask)
            output4 = next(layers)(output3, pos=pos, mask=mask)
            output = output + output1 + output2 + output3 + output4

            for layer in layers:
                output = layer(output, pos=pos, mask=mask)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, patch_dim, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.patch_dim = patch_dim

        # multihead attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None, mask=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        src2 = self.self_attn(q, k, src2)

        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
