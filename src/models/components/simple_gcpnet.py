import warnings
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F


from torch.nn import Linear,Sequential

from torch_scatter import scatter
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool,DiffGroupNorm

import numpy as np
from typing import Literal, Optional


from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv  import MessagePassing
from torch_geometric.utils    import softmax as tg_softmax
class RBFExpansion(nn.Module):

    def __init__(
        self,
        vmin: float = 0,
        vmax: float = 8,
        bins: int = 40,
        lengthscale: Optional[float] = None,
        type: str = "gaussian"
    ) -> None:
        """Initialize a `RBFExpansion` module.

        :param vmin: The minimum value of the input.
        :param vmax: The maximum value of the input.
        :param bins: The number of bins to use.
        :param lengthscale: The lengthscale of the RBF kernel.
        :param type: The type of RBF kernel to use.
        """
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(vmin, vmax, bins)
        )

        self.type = type


        if lengthscale is not None:
            self.lengthscale = lengthscale
            self.gamma = 1.0 / (lengthscale **2)
        else:
            self.lengthscale = torch.diff(self.centers).mean()
            self.gamma = 1.0 /self.lengthscale

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        base = self.gamma * (distance - self.centers)
        switcher = {
            'gaussian': (-base ** 2).exp(),
            'quadratic': base ** 2,
            'linear': base,
            'inverse_quadratic': 1.0 / (1.0 + base ** 2),
            'multiquadric': (1.0 + base ** 2).sqrt(),
            'inverse_multiquadric': 1.0 / (1.0 + base ** 2).sqrt(),
            'spline': base ** 2 * (base + 1.0).log(),
            'poisson_one': (base - 1.0) * (-base).exp(),
            'poisson_two': (base - 2.0) / 2.0 * base * (-base).exp(),
            'matern32': (1.0 + 3 ** 0.5 * base) * (-3 ** 0.5 * base).exp(),
            'matern52': (1.0 + 5 ** 0.5 * base + 5 / 3 * base ** 2) * (-5 ** 0.5 * base).exp(),
        }
        result = switcher.get(self.type, None)

        return result

class EmbeddingLayer(nn.Module):
    """
    Custom layer which performs nonlinear transform on embeddings

    Args:
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        activation (str, optional): Activation function to be applied. 
            Defaults to 'silu'.
        normalization (str, optional): Normalization method to be applied.
            Defaults to 'batch_norm'.
    """

    def __init__(self, input_features, output_features, activation='silu', normalization='batch_norm') -> None:
        super().__init__()
        # Define a Sequential model containing Linear layer, normalization, and activation
        self.mlp = nn.Sequential(
            nn.Linear(input_features, output_features),
            self.get_normalization(normalization, output_features),
            self.get_activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor.
        """
        return self.mlp(x)

    def get_activation(self, activation ):
        """
        Returns the activation function module based on the provided string.

        Args:
            activation (str): Activation function name.

        Returns:
            torch.nn.Module: Activation function module.
        """
        if activation.lower() == 'relu':
            return nn.ReLU()
        elif activation.lower() == 'leakyrelu':
            return nn.LeakyReLU()
        elif activation.lower() == 'sigmoid':
            return nn.Sigmoid()
        elif activation.lower() == 'tanh':
            return nn.Tanh()
        elif activation.lower() == 'silu':
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def get_normalization(self, normalization, num_features):
        """
        Returns the normalization layer module based on the provided string.

        Args:
            normalization (str): Normalization method name.
            num_features (int): Number of features.

        Returns:
            torch.nn.Module: Normalization layer module.
        """
        if normalization.lower() == 'batch_norm':
            return nn.BatchNorm1d(num_features)
        elif normalization.lower() == 'layer_norm':
            return nn.LayerNorm(num_features)
        elif normalization.lower() == 'instance_norm':
            return nn.InstanceNorm1d(num_features)
        else:
            raise ValueError(f"Unsupported normalization method: {normalization}")

class GCAO(MessagePassing):
    def __init__(self, dim, act='silu', batch_norm='False', batch_track_stats='False', dropout_rate=0.0, fc_layers=2, **kwargs):
        super(GCAO, self).__init__(aggr='add',flow='target_to_source', **kwargs)

        self.act          = act
        self.fc_layers    = fc_layers
        if batch_track_stats      == "False":
            self.batch_track_stats = False 
        else:
            self.batch_track_stats = True 

        self.batch_norm   = batch_norm
        self.dropout_rate = dropout_rate
 
        #####################  Graph Attention
        self.heads             = 4
        self.add_bias          = True
        self.neg_slope         = 0.2

        self.bn_node           = nn.BatchNorm1d(self.heads)
        self.W                 = Parameter(torch.Tensor(dim*2,self.heads*dim))
        self.att               = Parameter(torch.Tensor(1,self.heads,2*dim))
        self.dim               = dim

        if self.add_bias  : 
            self.bias = Parameter(torch.Tensor(dim))
        else              : 
            self.register_parameter('bias', None)
        ###################

        ###### Graph Convolution
        channels = dim # input vertex_dim
        if isinstance(channels, int):
            channels = (channels, channels) 
        # node * 2 + edge -> edge, since the embedding layer had map them into the same dimensional hidden_features, dim*3 -> dim is also OK
        self.lin_f = nn.Linear(sum(channels) + dim, dim, bias=True)
        self.lin_s = nn.Linear(sum(channels) + dim, dim, bias=True)
        if self.batch_norm:
            self.bn_edge = nn.BatchNorm1d(dim)    
        else:
            self.bn_edge = None

        ######
        self.reset_parameters() 
        # FIXED-lines -------------------------------------------------------------

    def reset_parameters(self):
        super().reset_parameters()
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)
        
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.batch_norm is not None:
            self.bn_edge.reset_parameters()
            self.bn_node.reset_parameters()


    def forward(self, node_feats, edge_index, edge_attr):
        node_feat = node_feats + self.propagate(edge_index, x=node_feats, edge_attr=edge_attr) # node attention
        edge_feat = edge_attr + self.update_edge(edge_index, node_feat, edge_attr) # edge convolution
        return node_feat, edge_feat

    def message(self, edge_index_i, x_i, x_j, edge_attr):
        '''
        node attention message
        '''
        node_i   = torch.cat([x_i,edge_attr],dim=-1)
        node_j   = torch.cat([x_j,edge_attr],dim=-1)
        
        node_i   = getattr(F, self.act)(torch.matmul(node_i,self.W))
        node_j   = getattr(F, self.act)(torch.matmul(node_j,self.W))
        node_i   = node_i.view(-1, self.heads, self.dim)
        node_j   = node_j.view(-1, self.heads, self.dim)

        alpha   = getattr(F, self.act)((torch.cat([node_i, node_j], dim=-1)*self.att).sum(dim=-1))
        alpha   = getattr(F, self.act)(self.bn_node(alpha))
        alpha   = tg_softmax(alpha,edge_index_i)

        alpha   = F.dropout(alpha, p=self.dropout_rate, training=self.training)
        node_j     = (node_j * alpha.view(-1, self.heads, 1)).transpose(0,1)

        return node_j

    def update(self, aggr_out):
        node = aggr_out.mean(dim=0)
        if self.bias is not None:  node = node + self.bias
        return node
    def update_edge(self, edge_index, x, edge_attr):
        '''edge convolution'''
        node_i        = x[edge_index[0]] # 节点i
        node_j        = x[edge_index[1]] # 节点j
        z   = torch.cat([node_i,node_j, edge_attr], dim=-1)
        message = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        message = message if self.batch_norm is None else self.bn_edge(message)
        return message



class GCPNetUpdate(torch.nn.Module):

    def __init__(self, hidden_features, dropout_rate) -> None:
        super().__init__()

        # Sequential GCPNet update layers
        # Overall mapping is input_features -> output_features
        self.bondAndAngleUpdate = GCAO(dim=hidden_features,dropout_rate=dropout_rate)
        self.bondAndAtomUpdate = GCAO(dim=hidden_features,dropout_rate=dropout_rate)

    def forward(
        self,
        g: Data,
        atom_feats: torch.Tensor,
        bond_attr: torch.Tensor,
        triplet_feats: torch.Tensor,
    ) -> torch.Tensor:
        # Perform sequential edge and node updates

        bond, triplet_feats = self.bondAndAngleUpdate(
            bond_attr,  g.angle_index, triplet_feats
        )

        atom_feats, bond_attr = self.bondAndAtomUpdate(atom_feats, g.edge_index, bond)

        # Return updated node, edge, and triplet embeddings
        return atom_feats, bond_attr, triplet_feats

class GCPNet(nn.Module):
    def __init__(self, 
                #  data: Data, 
                 firstUpdateLayers: int=4,
                 secondUpdateLayers: int=4,
                 atom_input_features: int=105,
                 edge_input_features: int=50,
                 triplet_input_features: int=40,
                 embedding_features: int=64,
                 hidden_features: int=256,
                 output_features: int=1,
                 min_edge_distance: float=0.0,
                 max_edge_distance: float=8.0,
                 min_angle: float=0.0,
                 max_angle: float=torch.acos(torch.zeros(1)).item() * 2,
                 link: Literal["identity", "log", "logit"] = "identity",
                 dropout_rate=0.0,
                ) -> None: 
        super().__init__()

        # self.data = data  ##Todo: for future utilization

        self.atom_embedding = EmbeddingLayer(atom_input_features, hidden_features)

        self.edge_embedding = torch.nn.Sequential(
            RBFExpansion(
                vmin=min_edge_distance, vmax=max_edge_distance, bins=edge_input_features
            ),
            EmbeddingLayer(edge_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )

        self.angle_embedding = torch.nn.Sequential(
            RBFExpansion(vmin=min_angle, vmax=max_angle, bins=triplet_input_features),
            EmbeddingLayer(triplet_input_features, embedding_features),
            EmbeddingLayer(embedding_features, hidden_features), 
        )

        # layer to perform atom, bond and andle updates on the graph by 2N GCAO
        self.firstUpdate = torch.nn.ModuleList(
            [GCPNetUpdate(hidden_features,dropout_rate) for _ in range(firstUpdateLayers)]
        )

        # layer to perform atom and bond updates on the graph by N GCAO
        self.secondUpdate = torch.nn.ModuleList(
            [
                GCAO(dim=hidden_features,dropout_rate=dropout_rate)
                for _ in range(secondUpdateLayers)
            ]
        )

        # simple hign-level output layer
        self.fc = Linear(hidden_features, output_features)

        switcher = {
            "identity": self._identity,
            "log": torch.exp,
            "logit": torch.sigmoid
        }
        self.link = switcher.get(link, None)
        if link == "log":
            avg_gap = 0.7
            self.fc.bias.data = torch.tensor(np.log(avg_gap), dtype=torch.float)
 
    def _identity(self, x):
        return x
    
    @property
    def target_attr(self):
        """Specifies the target attribute property for writing output to file"""
        return "y"

    def forward(self, g: Data) -> torch.Tensor:
        # unpack data
        atom_feats = g.x
        bond_attr = g.edge_attr
        triplet_feats = g.angle_attr

        # perform initial embedding
        atom_feats = self.atom_embedding(atom_feats)
        bond_attr = self.edge_embedding(bond_attr)
        triplet_feats = self.angle_embedding(triplet_feats)

        # perform sequential GCPNet updates
        for update in self.firstUpdate:
            atom_feats, bond_attr, triplet_feats = update(
                g, atom_feats, bond_attr, triplet_feats
            )

        for update in self.secondUpdate:
            atom_feats, bond_attr = update(
                atom_feats, g.edge_index, bond_attr
            )

        # for update in self.secondUpdate:
        #     out_nodes, _ = update(
        #         atom_feats, g.edge_index, bond_attr
        #     )
        #     atom_feats = atom_feats + out_nodes # skip connection on atom feats for each update

        # readout
        out = global_mean_pool(atom_feats, g.batch)
        out = self.fc(out)

        # apply link function
        if self.link:
            out = self.link(out)

        return out