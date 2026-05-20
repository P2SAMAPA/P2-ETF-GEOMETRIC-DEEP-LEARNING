"""config.py — Geometric Deep Learning Engine (E(n)-Equivariant GNN).

Core idea
---------
Treat each ETF as a point in R^d feature space. Build an E(n)-equivariant
graph neural network (EGNN, Satorras et al. 2021) over the ETF correlation
graph. The network is equivariant to rotations and translations of the
feature space — predictions are invariant to arbitrary orthogonal
transformations of the input features.

Why equivariance matters for ETFs
----------------------------------
Standard GNNs treat feature vectors as unordered bags of numbers.
EGNN respects the geometric structure of the feature space:
  - Rotation equivariance: rotating all ETF feature vectors produces
    a consistently rotated output — the ranking is preserved
  - Translation equivariance: shifting all features (e.g. a market-wide
    drift) doesn't change relative rankings
  - This makes predictions robust to arbitrary linear transformations
    of the input feature space — critical when features (returns, vol,
    macro) are measured in different units and scales

Architecture
------------
EGNN layer update:
  m_ij   = φ_e(h_i, h_j, ||x_i - x_j||², e_ij)   ← edge message
  x_i'   = x_i + C Σ_j (x_i - x_j) φ_x(m_ij)    ← equivariant coordinate update
  agg_i  = Σ_j φ_h(m_ij)                          ← message aggregation
  h_i'   = φ_h(h_i, agg_i)                         ← node feature update

x_i ∈ R^d = ETF position in feature space (coordinates)
h_i ∈ R^k = ETF hidden state (invariant features)
e_ij       = edge attributes (correlation strength)
"""

import os
from datetime import datetime

# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE   = "master_data.parquet"
HF_MODEL_REPO  = "P2SAMAPA/p2-etf-egnn-model"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-egnn-results"
HF_TOKEN       = os.environ.get("HF_TOKEN", None)

# ── Universes ─────────────────────────────────────────────────────────────────
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM",
]
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
COMBINED_TICKERS       = sorted(set(EQUITY_SECTORS_TICKERS + FI_COMMODITIES_TICKERS))

UNIVERSES = {
    "EQUITY_SECTORS":  EQUITY_SECTORS_TICKERS,
    "FI_COMMODITIES":  FI_COMMODITIES_TICKERS,
    "COMBINED":        COMBINED_TICKERS,
}

MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# ── Feature engineering ───────────────────────────────────────────────────────
# ETF coordinate space: each ETF is a point in R^COORD_DIM
# Built from: lagged returns + rolling vol + rolling momentum + macro sensitivities
COORD_DIM          = 8     # dimension of coordinate space x_i ∈ R^COORD_DIM
HIDDEN_DIM         = 64    # dimension of hidden state h_i
LOOKBACK_LAGS      = [1, 5, 21, 63]     # lagged return features
ROLLING_VOL_WINDOW = 21
ROLLING_MOM_WINDOW = 63

# ── Graph construction ────────────────────────────────────────────────────────
GRAPH_WINDOW       = 63     # rolling days for correlation graph
CORR_THRESHOLD     = 0.20   # minimum |correlation| to create edge
GRAPH_REFIT_FREQ   = 21     # refit correlation graph every N days

# ── EGNN architecture ─────────────────────────────────────────────────────────
N_EGNN_LAYERS      = 4      # number of EGNN message passing layers
MLP_HIDDEN         = 128    # hidden dim in edge/node MLPs
DROPOUT            = 0.10   # dropout rate
RESIDUAL           = True   # residual connections in node updates

# ── Training ──────────────────────────────────────────────────────────────────
TRAIN_END          = "2021-12-31"   # training cutoff
OOS_START          = "2022-01-01"   # OOS scoring start
LEARNING_RATE      = 1e-3
WEIGHT_DECAY       = 1e-4
N_EPOCHS           = 100
BATCH_SIZE         = 32             # number of time steps per batch
PATIENCE           = 15             # early stopping patience
GRAD_CLIP          = 1.0
TRAIN_WINDOW       = 252            # rolling training window days
REFIT_FREQ         = 63             # retrain model every N OOS days

# ── Scoring ───────────────────────────────────────────────────────────────────
CASH_THRESHOLD     = -0.40
TOP_N              = 6

# ── Checkpoint ───────────────────────────────────────────────────────────────
CKPT_MODEL         = "egnn_model_{slug}.pt"
CKPT_META          = "egnn_meta_{slug}.json"

TODAY = datetime.now().strftime("%Y-%m-%d")
