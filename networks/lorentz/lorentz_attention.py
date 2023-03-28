import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, repeat
from networks.lorentz.lorentz_utils import normsq4, dotsq4, psi

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class CoorsNorm(nn.Module):
    def __init__(self, eps = 1e-8, scale_init = 1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        return coors * self.scale

def exists(val):
    return val is not None

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

def batched_index_select(values, indices, dim = 1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)



class DynamicPositionBias(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads,
        depth,
        dim_head,
        input_dim = 1,
        norm = True
    ):
        super().__init__()
        assert depth >= 1, 'depth for dynamic position bias MLP must be greater or equal to 1'
        self.mlp = nn.ModuleList([])

        self.mlp.append(nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim) if norm else nn.Identity(),
            nn.SiLU()
        ))

        for _ in range(depth - 1):
            self.mlp.append(nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim) if norm else nn.Identity(),
                nn.SiLU()
            ))

        self.heads = heads
        self.qk_pos_head = nn.Linear(dim, heads)
        self.value_pos_head = nn.Linear(dim, dim_head * heads)

    def forward(self, pos):
        for layer in self.mlp:
            pos = layer(pos)

        qk_pos = self.qk_pos_head(pos)
        value_pos = self.value_pos_head(pos)

        qk_pos = rearrange(qk_pos, 'b 1 i j h -> b h i j')
        value_pos = rearrange(value_pos, 'b 1 i j (h d) -> b h i j d', h = self.heads)
        return qk_pos, value_pos


class LorentzAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 32,
        heads = 4,
        edge_dim = 0,
        coors_hidden_dim = 16,
        neighbors = 0,
        only_sparse_neighbors = False,
        valid_neighbor_radius = float('inf'),
        init_eps = 1e-3,
        dot_products = None,
        edge_mlp_mult = 2,
        norm_rel_coors = True,
        norm_coors_scale_init = 1.,
        talking_heads = False,
        scale = 8,
        lorentz_scale = True,
        dropout = 0.
    ):
        super().__init__()
        self.scale = scale
        self.lorentz_scale = lorentz_scale
        self.norm = LayerNorm(dim)

        self.neighbors = neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_neighbor_radius = valid_neighbor_radius

        attn_inner_dim = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Linear(dim, attn_inner_dim * 3, bias = False)
        self.to_out = nn.Linear(attn_inner_dim, dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else None

        self.edge_mlp = None
        has_edges = edge_dim > 0

        if has_edges:
            edge_input_dim = heads + edge_dim
            edge_hidden = edge_input_dim * edge_mlp_mult

            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden, bias = False),
                nn.GELU(),
                nn.Linear(edge_hidden, heads, bias = False)
            )

            self.coors_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(heads, heads, bias = False)
            )
        else:
            self.coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim, bias = False),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads, bias = False)
            )

        self.coors_gate = nn.Sequential(
            nn.Linear(heads, heads),
            nn.Tanh()
        )

        self.norm_rel_coors = CoorsNorm(scale_init = norm_coors_scale_init) if norm_rel_coors else nn.Identity()

        num_coors_combine_heads = 1 * heads
        self.coors_combine = nn.Parameter(torch.randn(num_coors_combine_heads))

        # positional embedding
        # for both along the sequence (if specified by rel_pos_emb) and the relative distance between each residue / atom

        self.dot_products = dot_products

        self.dynamic_pos_bias_mlp = DynamicPositionBias(
            dim = dim // 2,
            heads = heads,
            dim_head = dim_head,
            depth = 3,
            input_dim = (2 if dot_products else 1)
        )

        # dropouts

        self.node_dropout = nn.Dropout(dropout)
        self.coor_dropout = nn.Dropout(dropout)

        # init

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std = self.init_eps)

    def forward(
        self,
        feats,
        coors,
        edges = None,
        mask = None,
        adj_mat = None
    ):
        b, n, d, h, num_nn, only_sparse_neighbors, valid_neighbor_radius, device = *feats.shape, self.heads, self.neighbors, self.only_sparse_neighbors, self.valid_neighbor_radius, feats.device

        feats = self.norm(feats)

        assert not (only_sparse_neighbors and not exists(adj_mat)), 'adjacency matrix must be passed in if only_sparse_neighbors is turned on'

        if exists(mask):
            num_nodes = mask.sum(dim = -1)

        rel_coors_i, rel_coors_j = rearrange(coors, 'b i d -> b i () d'), rearrange(coors, 'b j d -> b () j d')
        rel_coors = rel_coors_i - rel_coors_j
        
        # rel_dist = rel_coors.norm(p = 2, dim = -1)

        
        rel_dist = normsq4(rel_coors)
        dot_products = dotsq4(rel_coors_i, rel_coors_j)
        if self.lorentz_scale:
            rel_dist = psi(rel_dist)
            dot_products = psi(dot_products)

        nbhd_indices = None
        nbhd_masks = None
        nbhd_ranking = rel_dist.clone()

        if exists(adj_mat):
            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat, 'i j -> b i j', b = b)

            self_mask = torch.eye(n, device = device).bool()
            self_mask = rearrange(self_mask, 'i j -> () i j')
            adj_mat.masked_fill_(self_mask, False)

            max_adj_neighbors = adj_mat.long().sum(dim = -1).max().item() + 1

            num_nn = max_adj_neighbors if only_sparse_neighbors else (num_nn + max_adj_neighbors)
            valid_neighbor_radius = 0 if only_sparse_neighbors else valid_neighbor_radius

            nbhd_ranking = nbhd_ranking.masked_fill(self_mask, -1.)
            nbhd_ranking = nbhd_ranking.masked_fill(adj_mat, 0.)

        if 0 < num_nn < n:
            print('num_nn', num_nn)
            # make sure padding does not end up becoming neighbors
            if exists(mask):
                ranking_mask = mask[:, :, None] * mask[:, None, :]
                nbhd_ranking = nbhd_ranking.masked_fill(~ranking_mask, 1e5)

            nbhd_values, nbhd_indices = nbhd_ranking.topk(num_nn, dim = -1, largest = False)
            nbhd_masks = nbhd_values <= valid_neighbor_radius

        # derive queries keys and values

        q, k, v = self.to_qkv(feats).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # calculate nearest neighbors

        i = j = n

        if exists(nbhd_indices):
            i, j = nbhd_indices.shape[-2:]
            nbhd_indices_with_heads = repeat(nbhd_indices, 'b n d -> b h n d', h = h)
            k         = batched_index_select(k, nbhd_indices_with_heads, dim = 2)
            v         = batched_index_select(v, nbhd_indices_with_heads, dim = 2)
            rel_dist  = batched_index_select(rel_dist, nbhd_indices, dim = 2)
            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim = 2)
        else:
            k = repeat(k, 'b h j d -> b h n j d', n = n)
            v = repeat(v, 'b h j d -> b h n j d', n = n)

        # prepare mask

        if exists(mask):
            q_mask = rearrange(mask, 'b i -> b 1 i 1')
            k_mask = repeat(mask, 'b j -> b i j', i = n)

            if exists(nbhd_indices):
                k_mask = batched_index_select(k_mask, nbhd_indices, dim = 2)

            k_mask = rearrange(k_mask, 'b i j -> b 1 i j')

            mask = q_mask * k_mask

            if exists(nbhd_masks):
                mask &= rearrange(nbhd_masks, 'b i j -> b 1 i j')

        # cosine sim attention

        q, k = map(l2norm, (q, k))

        # generate and apply rotary embeddings

        rel_dist = -rel_dist
        rel_dist = rearrange(rel_dist, 'b i j -> b 1 i j 1')

        if self.dot_products:
            dot_products = rearrange(dot_products, 'b i j -> b 1 i j 1')
            rel_dist = torch.cat((rel_dist, dot_products), dim = -1)

        qk_pos, value_pos = self.dynamic_pos_bias_mlp(rel_dist)

        # calculate inner product for queries and keys
        qk = einsum('b h i d, b h i j d -> b h i j', q, k) * (self.scale if not exists(edges) else 1)

        qk = qk + qk_pos

        v = v + value_pos

        # add edge information and pass through edges MLP if needed

        if exists(edges):
            if exists(nbhd_indices):
                edges = batched_index_select(edges, nbhd_indices, dim = 2)

            qk = rearrange(qk, 'b h i j -> b i j h')
            qk = torch.cat((qk, edges), dim = -1)
            qk = self.edge_mlp(qk)
            qk = rearrange(qk, 'b i j h -> b h i j')

        # coordinate MLP and calculate coordinate updates

        coors_mlp_input = rearrange(qk, 'b h i j -> b i j h')
        coor_weights = self.coors_mlp(coors_mlp_input)

        if exists(mask):
            mask_value = max_neg_value(coor_weights)
            coor_mask = repeat(mask, 'b 1 i j -> b i j 1')
            coor_weights.masked_fill_(~coor_mask, mask_value)

        coor_attn = coor_weights.softmax(dim = -2)
        coor_attn = self.coor_dropout(coor_attn)

        rel_coors_sign = self.coors_gate(coors_mlp_input)
        rel_coors_sign = rearrange(rel_coors_sign, 'b i j h -> b i j 1 h')

        rel_coors = self.norm_rel_coors(rel_coors)
        rel_coors = repeat(rel_coors, 'b i j c -> b i j c h', h = h)

        rel_coors = rel_coors * rel_coors_sign

        # aggregate and combine heads for coordinate updates

        rel_out = einsum('b i j h, b i j c h -> b i c h', coor_attn, rel_coors)

        coors_out = einsum('b n c h, h -> b n c', rel_out, self.coors_combine)

        # derive attention

        sim = qk.clone()

        if exists(mask):
            mask_value = max_neg_value(sim)
            sim.masked_fill_(~mask, mask_value)

        attn = sim.softmax(dim = -1)
        attn = self.node_dropout(attn)

        if exists(self.talking_heads):
            attn = self.talking_heads(attn)

        # weighted sum of values and combine heads

        out = einsum('b h i j, b h i j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, coors_out