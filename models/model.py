"""Hyperbolic Knowledge Graph embedding models where all parameters are defined in tangent spaces."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base import KGModel
from utils.hyperbolic import mobius_add, expmap0, project, hyp_distance_multi_c, logmap0

MODELS = ["HAQE"]


class BaseHAQE(KGModel):
    """Trainable curvature for each relationship."""

    def __init__(self, args):
        super(BaseHAQE, self).__init__(args.sizes, args.rank, args.dropout, args.gamma, args.dtype, args.bias,
                                    args.init_size)
        self.entity.weight.data = self.init_size * torch.randn((self.sizes[0], 4 * self.rank), dtype=self.data_type)
        self.rel.weight.data = self.init_size * torch.randn((self.sizes[1], 2 * 4 * self.rank), dtype=self.data_type)
        self.rel_diag = nn.Embedding(self.sizes[1], 4 * self.rank, dtype=self.data_type)
        # self.rel_diag.weight.data = self.init_size * torch.randn((self.sizes[1], 4 * self.rank), dtype=self.data_type)

        self.rel_diag.weight.data = 2 * torch.rand((self.sizes[1], 4 * self.rank), dtype=self.data_type) - 1.0

        self.multi_c = args.multi_c
        if self.multi_c:
            c_init = torch.ones((self.sizes[1], 1), dtype=self.data_type)
        else:
            c_init = torch.ones((1, 1), dtype=self.data_type)
        self.c = nn.Parameter(c_init, requires_grad=True)

    def get_rhs(self, queries, eval_mode):
        """Get embeddings and biases of target entities."""
        if eval_mode:
            return self.entity.weight, self.bt.weight
        else:
            return self.entity(queries[:, 2]), self.bt(queries[:, 2])

    def similarity_score(self, lhs_e, rhs_e, eval_mode):
        """Compute similarity scores or queries against targets in embedding space."""
        lhs_e, c = lhs_e
        return - hyp_distance_multi_c(lhs_e, rhs_e, c, eval_mode) ** 2

    # def get_factors(self, queries):
    #     """Compute factors for embeddings' regularization."""
    #     lhs = self.entity(queries[:, 0])
    #     rel = self.rel(queries[:, 1])
    #     rhs = self.entity(queries[:, 2])
    #
    #     head_e = torch.chunk(lhs, 4, dim=1)
    #     rel_e = torch.chunk(rel, 4, dim=1)
    #     rhs_e = torch.chunk(rhs, 4, dim=1)
    #
    #     head_f = torch.sqrt(head_e[0] ** 2 + head_e[1] ** 2 + head_e[2] ** 2 + head_e[3] ** 2)
    #     rel_f = torch.sqrt(rel_e[0] ** 2 + rel_e[1] ** 2 + rel_e[2] ** 2 + rel_e[3] ** 2)
    #     rhs_f = torch.sqrt(rhs_e[0] ** 2 + rhs_e[1] ** 2 + rhs_e[2] ** 2 + rhs_e[3] ** 2)
    #
    #     return head_f, rel_f, rhs_f


class HAQE(BaseHyperQE):

    def _cal(self, h, r):
        s_a, x_a, y_a, z_a = torch.chunk(h, 4, dim=1)
        s_b, x_b, y_b, z_b = torch.chunk(r, 4, dim=1)
        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        return torch.cat([A, B, C, D], dim=1)

    def get_queries(self, queries):
        """Compute embedding and biases of queries."""
        c = F.softplus(self.c[queries[:, 1]])
        head = expmap0(self.entity(queries[:, 0]), c)
        rel1, rel2 = torch.chunk(self.rel(queries[:, 1]), 2, dim=1)
        rel1 = expmap0(rel1, c)
        rel2 = expmap0(rel2, c)
        lhs = logmap0(mobius_add(head, rel1, c), c)
        rel = self.rel_diag(queries[:, 1])
        res1 = self._cal(lhs, rel)
        res1 = expmap0(res1, c)
        res2 = mobius_add(res1, rel2, c)
        return (res2, c), self.bh(queries[:, 0])
