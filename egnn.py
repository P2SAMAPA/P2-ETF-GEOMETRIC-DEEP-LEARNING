"""egnn.py — E(n)-Equivariant Graph Neural Network (Satorras et al. 2021).

EGNN update equations per layer
--------------------------------
Edge message:
    m_ij = φ_e(h_i, h_j, ||x_i - x_j||², e_ij)

Equivariant coordinate update:
    x_i' = x_i + C Σ_{j≠i} (x_i - x_j) φ_x(m_ij)

Node feature update:
    agg_i = Σ_j φ_a(m_ij)
    h_i'  = φ_h(h_i, agg_i)

Properties
----------
- E(n)-equivariant: rotating/translating all coordinates x_i
  produces consistently rotated/translated output coordinates x_i'
- Invariant node features h_i are unaffected by coordinate transforms
- Distances ||x_i - x_j||² are rotation/translation invariant
- Final score readout uses only h_i (invariant) → score is invariant
  to all orthogonal transformations of the feature space

This is strictly more principled than CLIFFORD-NET (Clifford algebra
grade-level equivariance) for the case of coordinate-space rotational
equivariance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.SiLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


class EGNNLayer(nn.Module):
    """Single E(n)-equivariant message passing layer.

    Parameters
    ----------
    h_dim     : dimension of invariant node features
    coord_dim : dimension of equivariant coordinates (= config.COORD_DIM)
    edge_dim  : dimension of edge attributes
    hidden    : hidden dim in edge/node MLPs
    residual  : add residual connection for h updates
    """

    def __init__(
        self,
        h_dim:     int,
        coord_dim: int,
        edge_dim:  int  = 1,
        hidden:    int  = config.MLP_HIDDEN,
        dropout:   float = config.DROPOUT,
        residual:  bool  = config.RESIDUAL,
    ) -> None:
        super().__init__()
        self.residual  = residual
        self.coord_dim = coord_dim

        # φ_e: edge message network
        # Input: h_i ⊕ h_j ⊕ ||x_i-x_j||² ⊕ e_ij
        self.phi_e = _mlp(h_dim * 2 + 1 + edge_dim, hidden, hidden, dropout)

        # φ_x: coordinate update weight (scalar per edge)
        self.phi_x = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        # φ_h: node feature update
        # Input: h_i ⊕ agg_i
        self.phi_h = _mlp(h_dim + hidden, hidden, h_dim, dropout)

        # Normalisation constant C (learnable)
        self.C = nn.Parameter(torch.ones(1) * 0.1)

    def forward(
        self,
        h:          torch.Tensor,       # (N, h_dim) invariant node features
        x:          torch.Tensor,       # (N, coord_dim) equivariant coordinates
        edge_index: torch.Tensor,       # (2, E) edges [src, dst]
        edge_attr:  torch.Tensor,       # (E, edge_dim) edge attributes
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass. Returns updated (h', x')."""
        src, dst = edge_index[0], edge_index[1]
        N = h.size(0)

        # ── Edge messages ─────────────────────────────────────────────────────
        x_diff  = x[src] - x[dst]                          # (E, coord_dim)
        sq_dist = (x_diff ** 2).sum(dim=-1, keepdim=True)  # (E, 1)  ← invariant

        edge_inp = torch.cat([h[src], h[dst], sq_dist, edge_attr], dim=-1)
        m_ij     = self.phi_e(edge_inp)                     # (E, hidden)

        # ── Coordinate update (equivariant) ──────────────────────────────────
        # w_ij = scalar weight; x update = Σ_j w_ij * (x_i - x_j)
        w_ij   = self.phi_x(m_ij)                           # (E, 1)
        # Scatter weighted direction vectors onto source nodes
        coord_update = torch.zeros_like(x)                  # (N, coord_dim)
        coord_update.scatter_add_(
            0,
            src.unsqueeze(1).expand(-1, self.coord_dim),
            w_ij * x_diff,
        )
        x_new = x + self.C * coord_update                   # (N, coord_dim)

        # ── Node feature update (invariant) ───────────────────────────────────
        agg = torch.zeros(N, m_ij.size(-1), device=h.device)
        agg.scatter_add_(0, src.unsqueeze(1).expand_as(m_ij), m_ij)

        h_inp = torch.cat([h, agg], dim=-1)                 # (N, h_dim+hidden)
        h_new = self.phi_h(h_inp)                           # (N, h_dim)
        if self.residual:
            h_new = h_new + h

        return h_new, x_new


class EGNNModel(nn.Module):
    """Full E(n)-equivariant GNN for ETF scoring.

    Pipeline:
      1. Project raw node features → h_dim invariant hidden state
      2. Project raw coordinates → coord_dim equivariant space
      3. N_EGNN_LAYERS of equivariant message passing
      4. Readout: MLP(h_final) → scalar score per ETF

    The readout uses only h (invariant features) so the final
    ETF scores are invariant to any rotation/translation of the
    input coordinate space.
    """

    def __init__(
        self,
        in_dim:    int,      # raw feature dimension per node
        coord_dim: int = config.COORD_DIM,
        h_dim:     int = config.HIDDEN_DIM,
        n_layers:  int = config.N_EGNN_LAYERS,
        edge_dim:  int = 1,
        hidden:    int = config.MLP_HIDDEN,
        dropout:   float = config.DROPOUT,
    ) -> None:
        super().__init__()
        self.coord_dim = coord_dim

        # Input projections
        self.node_embed  = nn.Linear(in_dim, h_dim)
        self.coord_embed = nn.Linear(in_dim, coord_dim)

        # EGNN layers
        self.layers = nn.ModuleList([
            EGNNLayer(h_dim, coord_dim, edge_dim, hidden, dropout)
            for _ in range(n_layers)
        ])

        # Readout: invariant score per ETF
        self.readout = nn.Sequential(
            nn.Linear(h_dim, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        node_feats: torch.Tensor,    # (N, in_dim)
        edge_index: torch.Tensor,    # (2, E)
        edge_attr:  torch.Tensor,    # (E, edge_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns
        -------
        scores : (N,)         — raw scalar score per ETF (invariant)
        h_out  : (N, h_dim)   — final node features (invariant)
        x_out  : (N, coord_dim) — final coordinates (equivariant)
        """
        # Initial embeddings
        h = F.silu(self.node_embed(node_feats))    # (N, h_dim)
        x = self.coord_embed(node_feats)            # (N, coord_dim) — initial coords

        # EGNN message passing
        for layer in self.layers:
            h, x = layer(h, x, edge_index, edge_attr)

        # Readout (invariant)
        scores = self.readout(h).squeeze(-1)        # (N,)
        return scores, h, x
