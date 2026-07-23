#!/usr/bin/env python3

# purpose: holds the 2 matrices and bias and gets updated often

import numpy as np
import torch
import torch.nn as nn

FEAT_DIM = 44
HIDDEN = 128


class BiaffineEdgeScorer(nn.Module):
    def __init__(self, feat_dim=FEAT_DIM, hidden=HIDDEN):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.src_proj = nn.Linear(feat_dim, hidden, bias=False)
        self.dst_proj = nn.Linear(feat_dim, hidden, bias=False)
        self.bias = nn.Parameter(torch.zeros(1))

    # gives 2 matrices
    # matrix 1 is a dot product of feature weights and W_src
    # gives a num_nodes x 128 sized matrix
    # matrix 2 is same, but multiplied feature weights with W_dst
    def node_proj(self, feats):
        return self.src_proj(feats), self.dst_proj(feats)

    # gives a score per edge by taking the two rows in their src and dest matrices respectively and do dot product 
    # add bias
    # get score
    def edge_scores(self, feats, src, dst):
        s, d = self.node_proj(feats)
        return (s[src] * d[dst]).sum(-1) + self.bias

    def forward(self, feats, src, dst):
        return self.edge_scores(feats, src, dst)

    # export trained matrices as numpy arrays (use at the very end)
    def to_numpy(self):
        return {
            "src_proj": self.src_proj.weight.detach().cpu().numpy().astype(np.float32),
            "dst_proj": self.dst_proj.weight.detach().cpu().numpy().astype(np.float32),
            "bias": self.bias.detach().cpu().numpy().astype(np.float32),
        }

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
