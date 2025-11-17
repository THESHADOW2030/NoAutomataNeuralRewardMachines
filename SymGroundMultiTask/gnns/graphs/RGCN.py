import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch.conv import RelGraphConv

from SymGroundMultiTask.gnns.graphs.GNN import GNN
from SymGroundMultiTask.utils.ast_builder import edge_types



class RGCN(GNN):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim)

        hidden_dims = kwargs.get('hidden_dims', [32])
        self.num_layers = len(hidden_dims)

        hidden_plus_input_dims = [hd + input_dim for hd in hidden_dims]
        self.convs = nn.ModuleList([RelGraphConv(in_dim, out_dim, len(edge_types), activation=F.relu)
            for (in_dim, out_dim) in zip([input_dim] + hidden_plus_input_dims[:-1], hidden_dims)])

        self.g_embed = nn.Linear(hidden_dims[-1], output_dim)


    def forward(self, g):

        g = dgl.batch(g.squeeze(1), ndata=['feat', 'is_root'])
        h_0 = g.ndata["feat"].float()

        h = h_0
        for i in range(self.num_layers):
            if i != 0:
                h = self.convs[i](g, torch.cat([h, h_0], dim=1), g.edata["type"])
            else:
                h = self.convs[i](g, h, g.edata["type"])
        g.ndata['h'] = h

        # Calculate graph representation by averaging all the hidden node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.g_embed(hg).squeeze(1)



class RGCNRoot(RGCN):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)


    def forward(self, g):

        g = dgl.batch(g.squeeze(1), ndata=['feat', 'is_root'])
        h_0 = g.ndata["feat"].float()

        h = h_0
        for i in range(self.num_layers):
            if i != 0:
                h = self.convs[i](g, torch.cat([h, h_0], dim=1), g.edata["type"])
            else:
                h = self.convs[i](g, h, g.edata["type"])
        g.ndata['h'] = h

        hg = g.ndata['h'][g.ndata['is_root'].bool()]
        return self.g_embed(hg).squeeze(1)


# Root: the final representation depends only on the last-layer embedding of the root node
# Shared: all the RGCN layers use the same weights
# Uses initial residual connection: input of each layer is concatenation of initial input and current representation
class RGCNRootShared(GNN):

    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim)
        hidden_dim = kwargs.get('hidden_dim', 32)
        num_layers = kwargs.get('num_layers', 2)

        self.num_layers = num_layers
        self.linear_in = nn.Linear(input_dim, hidden_dim)
        self.conv = RelGraphConv(2*hidden_dim, hidden_dim, len(edge_types), activation=torch.tanh)
        self.g_embed = nn.Linear(hidden_dim, output_dim)


    def forward(self, g):

        g = dgl.batch(g.squeeze(1), ndata=['feat', 'is_root'])
        h_0 = self.linear_in(g.ndata["feat"].float())

        h = h_0
        for _ in range(self.num_layers):
            h = self.conv(g, torch.cat([h, h_0], dim=1), g.edata["type"])
        g.ndata['h'] = h

        hg = g.ndata['h'][g.ndata['is_root'].bool()]
        return self.g_embed(hg).squeeze(1)