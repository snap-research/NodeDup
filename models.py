import torch
import torch.nn as nn

# from torch_geometric.nn import GCNConv, SAGEConv, GATConv, APPNP
from torch_geometric.nn import SAGEConv, GATConv, APPNP, GCNConv
import torch.nn.functional as F
from Conv import Sage_conv
# from gcnconv import GCNConv

import torch.nn.utils.prune as prune

class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
        cold_dropout=False,
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = dropout_ratio
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.cold_dropout = cold_dropout

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, feats, data=None):
        x = feats
        iso_p, cold_p, warm_p = dropout_p(self.dropout)
        for l, layer in enumerate(self.layers):
            x = layer(x)
            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    x = self.norms[l](x)
                x = F.relu(x)
                if self.cold_dropout:
                    if self.training:
                        x_iso = x[data.ISO_mask]
                        x_cold = x[data.COLD_mask]
                        x_warm = x[data.WARM_mask]
                        x = x.new_empty(x.size())
                        # assert x_iso.size(0) + x_cold.size(0) + x_warm.size(0) == x.size(0)
                        x_iso = F.dropout(x_iso, p=iso_p, training=self.training)
                        x_cold = F.dropout(x_cold, p=cold_p, training=self.training)
                        x_warm = F.dropout(x_warm, p=warm_p, training=self.training)
                        x[data.ISO_mask] = x_iso
                        x[data.COLD_mask] = x_cold
                        x[data.WARM_mask] = x_warm
                else:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, cold_dropout=False):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=True))
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))

        self.dropout = dropout
        self.cold_dropout = cold_dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, data=None):
        iso_p, cold_p, warm_p = dropout_p(self.dropout)
        for conv in self.convs[:-1]:
            # import ipdb; ipdb.set_trace()
            x = conv(x, adj_t)
            x = F.relu(x)
            if self.cold_dropout:
                if self.training:
                    x_iso = x[data.ISO_mask]
                    x_cold = x[data.COLD_mask]
                    x_warm = x[data.WARM_mask]
                    x = x.new_empty(x.size())
                    # assert x_iso.size(0) + x_cold.size(0) + x_warm.size(0) == x.size(0)
                    x_iso = F.dropout(x_iso, p=iso_p, training=self.training)
                    x_cold = F.dropout(x_cold, p=cold_p, training=self.training)
                    x_warm = F.dropout(x_warm, p=warm_p, training=self.training)
                    x[data.ISO_mask] = x_iso
                    x[data.COLD_mask] = x_cold
                    x[data.WARM_mask] = x_warm
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

def dropout_p(p):
    iso_p = p**3
    cold_p = (p+iso_p)/3
    return torch.FloatTensor([iso_p, cold_p, p])

class JKNet(torch.nn.Module):
    def __init__(self, dataset, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, mode='cat'):
        super(JKNet, self).__init__()
        from torch_geometric.nn import JumpingKnowledge
        self.num_layers = num_layers
        self.mode = mode

        if dataset == "coauthor-physics" or dataset == "igb-tiny" or dataset == "igb-small":
            self.conv0 = Sage_conv(in_channels, hidden_channels)
        else:
            self.conv0 = SAGEConv(in_channels, hidden_channels)
        self.dropout0 = nn.Dropout(p=dropout)

        for i in range(1, self.num_layers):
            if dataset == "coauthor-physics" or dataset == "igb-tiny" or dataset == "igb-small":
                setattr(self, 'conv{}'.format(i), Sage_conv(hidden_channels, hidden_channels))
            else:
                setattr(self, 'conv{}'.format(i), SAGEConv(hidden_channels, hidden_channels))
            setattr(self, 'dropout{}'.format(i), nn.Dropout(p=dropout))

        self.jk = JumpingKnowledge(mode=mode)
        if mode == 'max':
            self.fc = nn.Linear(hidden_channels, out_channels)
        elif mode == 'cat':
            self.fc = nn.Linear(num_layers * hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.jk.reset_parameters()
        self.fc.reset_parameters()

    def forward(self, x, adj_t, data=None):
        x, edge_index = x, adj_t

        layer_out = []  # 保存每一层的结果
        for i in range(self.num_layers):
            conv = getattr(self, 'conv{}'.format(i))
            dropout = getattr(self, 'dropout{}'.format(i))
            x = dropout(F.relu(conv(x, edge_index)))
            layer_out.append(x)

        h = self.jk(layer_out)  # JK层

        h = self.fc(h)

        return h

# class SAGE(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
#                  dropout):
#         super(SAGE, self).__init__()

#         self.convs = torch.nn.ModuleList()
#         self.convs.append(SAGEConv(in_channels, hidden_channels))
#         for _ in range(num_layers - 2):
#             self.convs.append(SAGEConv(hidden_channels, hidden_channels))
#         self.convs.append(SAGEConv(hidden_channels, out_channels))

#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()

#     def forward(self, x, adj_t):
#         for conv in self.convs[:-1]:
#             x = conv(x, adj_t)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
#         return x

class SAGE(torch.nn.Module):
    def __init__(self, data_name, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, norm_type="none", cold_dropout=False):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.norms = nn.ModuleList()
        self.norm_type = norm_type
        self.cold_dropout=cold_dropout
        if self.norm_type == "batch":
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        elif self.norm_type == "layer":
            self.norms.append(nn.LayerNorm(hidden_channels))            

        if data_name == "coauthor-physics" or data_name == "igb-tiny" or data_name == "igb-small":
            self.convs.append(Sage_conv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(Sage_conv(hidden_channels, hidden_channels))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_channels))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_channels))
            self.convs.append(Sage_conv(hidden_channels, out_channels))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_channels))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_channels))
            self.convs.append(SAGEConv(hidden_channels, out_channels))

        # for conv in self.convs:
        #     prune.random_unstructured(conv.lin_l, name="weight", amount=0.3)

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, data=None):
        iso_p, cold_p, warm_p = dropout_p(self.dropout)
        for l, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            if self.norm_type != "none":
                    x = self.norms[l](x)
            x = F.relu(x)
            if self.cold_dropout:
                if self.training:
                    x_iso = x[data.ISO_mask]
                    x_cold = x[data.COLD_mask]
                    x_warm = x[data.WARM_mask]
                    x = x.new_empty(x.size())
                    # assert x_iso.size(0) + x_cold.size(0) + x_warm.size(0) == x.size(0)
                    x_iso = F.dropout(x_iso, p=iso_p, training=self.training)
                    x_cold = F.dropout(x_cold, p=cold_p, training=self.training)
                    x_warm = F.dropout(x_warm, p=warm_p, training=self.training)
                    x[data.ISO_mask] = x_iso
                    x[data.COLD_mask] = x_cold
                    x[data.WARM_mask] = x_warm
            else:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x

        # for l, conv in enumerate(self.convs[:-1]):
        #     x = conv(x, adj_t)
        #     if self.norm_type != "none":
        #             x = self.norms[l](x)
        #     x = F.relu(x)
        #     x = F.dropout(x, p=self.dropout, training=self.training)
        # x = self.convs[-1](x, adj_t)
        # return x

class APPNP_model(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, norm_type="none", alpha=0.1, k=10):
        super(APPNP_model, self).__init__()

        self.num_layers = num_layers
        self.norm_type = norm_type
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            if self.norm_type == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            elif self.norm_type == "layer":
                self.norms.append(nn.LayerNorm(hidden_dim))

            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                if self.norm_type == "batch":
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
                elif self.norm_type == "layer":
                    self.norms.append(nn.LayerNorm(hidden_dim))

            self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.propagate = APPNP(k, alpha, 0.)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj_t):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(h)

            if l != self.num_layers - 1:
                if self.norm_type != "none":
                    h = self.norms[l](h)
                h = F.relu(h)
                h = self.dropout(h)

        h = self.propagate(h, adj_t)
        return h

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, dropout, norm_type="none"):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout
        self.convs.append(GATConv(in_channels, hidden_channels, heads, dropout=self.dropout))
        # On the Pubmed dataset, use `heads` output heads in `conv2`.
        self.convs.append(GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=self.dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t, data=None):
        for l, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, predictor, in_channels, hidden_channels, out_channels, num_layers,
                 dropout):
        super(LinkPredictor, self).__init__()

        self.predictor = predictor
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.predictor == 'mlp':
            for lin in self.lins[:-1]:
                x = lin(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.lins[-1](x)
        elif self.predictor == 'mean':
            x = torch.mean(x, dim=-1)
        else:
            x = torch.sum(x, dim=-1)

        return x