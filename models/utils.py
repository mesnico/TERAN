import math

import torch
from torch import nn, nn as nn


class PositionalEncodingText(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncodingText, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PositionalEncodingImageGrid(nn.Module):
    def __init__(self, d_model, n_regions=(4, 4)):
        super().__init__()
        assert n_regions[0] == n_regions[1]
        self.map = nn.Linear(2, d_model)
        self.n_regions = n_regions
        self.coord_tensor = self.build_coord_tensor(n_regions[0])

    @staticmethod
    def build_coord_tensor(d):
        coords = torch.linspace(-1., 1., d)
        x = coords.unsqueeze(0).repeat(d, 1)
        y = coords.unsqueeze(1).repeat(1, d)
        ct = torch.stack((x, y), dim=2)
        if torch.cuda.is_available():
            ct = ct.cuda()
        return ct

    def forward(self, x, start_token=False):   # x is seq_len x B x dim
        assert not (start_token and self.n_regions[0] == math.sqrt(x.shape[0]))
        bs = x.shape[1]
        ct = self.coord_tensor.view(self.n_regions[0]**2, -1)   # 16 x 2

        ct = self.map(ct).unsqueeze(1)   # 16 x d_model
        if start_token:
            x[1:] = x[1:] + ct.expand(-1, bs, -1)
            out_grid_point = torch.FloatTensor([-1. - 2/self.n_regions[0], -1.]).unsqueeze(0)
            if torch.cuda.is_available():
                out_grid_point = out_grid_point.cuda()
            x[0:1] = x[0:1] + self.map(out_grid_point)
        else:
            x = x + ct.expand(-1, bs, -1)
        return x


class PositionalEncodingImageBoxes(nn.Module):
    def __init__(self, d_model, mode='project-and-sum'):
        super().__init__()
        self.mode = mode
        if mode == 'project-and-sum':
            self.map = nn.Linear(5, d_model)
        elif mode == 'concat-and-process':
            self.map = nn.Sequential(
                nn.Linear(d_model + 5, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )


    def forward(self, x, boxes):  # x is seq_len x B x dim
        bs = x.shape[1]
        area = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
        area = area.unsqueeze(2)
        s_infos = torch.cat([boxes, area], dim=2)
        if self.mode == 'project-and-sum':
            ct = self.map(s_infos).permute(1, 0, 2)    # S x B x dim
            x = x + ct.expand(-1, bs, -1)
        elif self.mode == 'concat-and-process':
            x = torch.cat([x, s_infos.permute(1, 0, 2)], dim=2)
            x = self.map(x)
        return x


def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


class GatedAggregation(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.gate_fn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1)
        )
        self.node_fn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, x, mask):
        out = x.permute(1, 0, 2)
        gate = self.gate_fn(out)
        gate = gate.masked_fill_(mask.unsqueeze(2), - float('inf'))
        m = torch.sigmoid(gate)  # B x S x 1
        v = self.node_fn(out)  # B x S x dim
        out = torch.bmm(m.permute(0, 2, 1), v)  # B x 1 x dim
        out = out.squeeze(1)  # B x dim
        return out


class Aggregator(nn.Module):
    def __init__(self, embed_size, aggregation_type='sum'):
        super().__init__()
        self.aggregation = aggregation_type
        if self.aggregation == 'gated':
            self.gated_aggr = GatedAggregation(embed_size)
        if self.aggregation == 'gru':
            self.gru_aggr = nn.GRU(embed_size, embed_size, batch_first=True)
        if self.aggregation == 'sum-and-map':
            self.map = nn.Sequential(
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.Linear(embed_size, embed_size)
            )

    def forward(self, x, lengths, mask):
        if self.aggregation == 'first':
            out = x[0, :, :]
        elif self.aggregation == 'sum':
            x = x.permute(1, 0, 2)
            for o, c_len in zip(x, lengths):
                o[c_len:] = 0
            out = x.sum(dim=1)
        elif self.aggregation == 'gated':
            out = self.gated_aggr(x, mask)
        elif self.aggregation == 'gru':
            packed_sequence = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=False, enforce_sorted=False)
            _, out = self.gru_aggr(packed_sequence)
            out = out.squeeze(0)
        elif self.aggregation == 'sum-and-map':
            x = x.permute(1, 0, 2)
            for o, c_len in zip(x, lengths):
                o[c_len:] = 0
            out = x.sum(dim=1)
            out = self.map(out)
        else:
            raise ValueError('Final aggregation not defined!')

        return out


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask