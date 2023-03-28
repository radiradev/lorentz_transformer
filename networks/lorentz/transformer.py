
import torch
from torch import nn
from einops import repeat
import torch.nn.functional as F

from networks.lorentz.lorentz_attention import exists, default
from en_transformer.en_transformer import Residual
from torch.utils.checkpoint import checkpoint_sequential
from networks.lorentz.lorentz_attention import LorentzAttention

class Block(nn.Module):
    def __init__(self, attn, ff):
        super().__init__()
        self.attn = attn
        self.ff = ff

    def forward(self, inp, coor_changes = None):
        feats, coors, mask, edges, adj_mat = inp
        feats, coors = self.attn(feats, coors, edges = edges, mask = mask, adj_mat = adj_mat)
        feats, coors = self.ff(feats, coors)
        return (feats, coors, mask, edges, adj_mat)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = int(dim * mult * 2 / 3)

        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2, bias = False),
            GEGLU(),
            LayerNorm(inner_dim),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(self, feats, coors):
        return self.net(feats), 0


class LorentzTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        lorentz_scale=True,
        dot_products=True,
        num_tokens = None,
        rel_pos_emb = False,
        dim_head = 64,
        heads = 8,
        num_edge_tokens = None,
        edge_dim = 0,
        coors_hidden_dim = 16,
        neighbors = 0,
        only_sparse_neighbors = False,
        num_adj_degrees = None,
        adj_dim = 0,
        valid_neighbor_radius = float('inf'),
        init_eps = 1e-3,
        norm_rel_coors = True,
        norm_coors_scale_init = 1.,
        talking_heads = False,
        checkpoint = False,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        assert dim_head >= 32, 'your dimension per head should be greater than 32 for rotary embeddings to work well'
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'

        if only_sparse_neighbors:
            num_adj_degrees = default(num_adj_degrees, 1)

        self.token_emb = nn.Embedding(num_tokens, dim) if exists(num_tokens) else None
        self.edge_emb = nn.Embedding(num_edge_tokens, edge_dim) if exists(num_edge_tokens) else None

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        self.checkpoint = checkpoint
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            self.layers.append(Block(
                Residual(LorentzAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    coors_hidden_dim = coors_hidden_dim,
                    edge_dim = (edge_dim + adj_dim),
                    neighbors = neighbors,
                    only_sparse_neighbors = only_sparse_neighbors,
                    valid_neighbor_radius = valid_neighbor_radius,
                    init_eps = init_eps,
                    norm_rel_coors = norm_rel_coors,
                    norm_coors_scale_init = norm_coors_scale_init,
                    talking_heads = talking_heads,
                    dropout = attn_dropout,
                    dot_products=dot_products,
                    lorentz_scale=lorentz_scale,

                )),
                Residual(FeedForward(
                    dim = dim,
                    dropout = ff_dropout
                ))
            ))

    def forward(
        self,
        feats,
        coors,
        edges = None,
        mask = None,
        adj_mat = None,
        return_coor_changes = False,
        **kwargs
    ):
        b = feats.shape[0]

        if exists(self.token_emb):
            feats = self.token_emb(feats)

        if exists(self.edge_emb):
            assert exists(edges), 'edges must be passed in as (batch x seq x seq) indicating edge type'
            edges = self.edge_emb(edges)

        assert not (exists(adj_mat) and (not exists(self.num_adj_degrees) or self.num_adj_degrees == 0)), 'num_adj_degrees must be greater than 0 if you are passing in an adjacency matrix'

        if exists(self.num_adj_degrees):
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        assert not (return_coor_changes and self.training), 'you must be eval mode in order to return coordinates'

        # go through layers

        coor_changes = [coors]
        inp = (feats, coors, mask, edges, adj_mat)

        # if in training mode and checkpointing is designated, use checkpointing across blocks to save memory

        if self.training and self.checkpoint:
            inp = checkpoint_sequential(self.layers, len(self.layers), inp)
        else:
            # iterate through blocks
            for layer in self.layers:
                inp = layer(inp)
                coor_changes.append(inp[1]) # append coordinates for visualization

        # return

        feats, coors, *extra = inp

        if return_coor_changes:
            return feats, coors, coor_changes

        return feats, coors