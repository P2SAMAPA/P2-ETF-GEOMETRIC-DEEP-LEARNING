# 🔷 P2-ETF-GEOMETRIC-DEEP-LEARNING

**P2Quant Engine** · E(n)-Equivariant Graph Neural Network · Geometric Deep Learning · ETF Ranking

[![EGNN Daily Inference](https://github.com/P2SAMAPA/P2-ETF-GEOMETRIC-DEEP-LEARNING/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-GEOMETRIC-DEEP-LEARNING/actions/workflows/daily_run.yml)

---

## What Is This?

This engine treats each ETF as a **point in equivariant coordinate space** and applies
**E(n)-equivariant graph neural networks** (EGNN, Satorras et al. 2021) over the ETF
correlation graph. Unlike all other graph engines in the suite (GCN, GAT, Hypergraph,
Graph Transformer), EGNN is provably equivariant to rotations and translations of the
feature space — the ranking is invariant to any orthogonal transformation of inputs.

---

## Why Equivariance Matters for ETF Ranking

Standard GNNs treat feature vectors as unordered bags of numbers. If you rescale
VIX from points to z-scores, or rotate the feature coordinate system, a standard GNN
produces different (arbitrary) outputs.

**EGNN is provably invariant:** rotating or translating all coordinate vectors x_i
produces exactly the same invariant scores. This means:

- Rankings are stable across different feature normalisations and scales
- Distances ||x_i − x_j||² between ETFs are always invariant (rotation/translation proof)
- The equivariant coordinate updates encode genuine geometric structure, not scale artefacts

---

## EGNN Update Equations (per layer)

```
Edge message:
    m_ij   = φ_e(h_i, h_j, ||x_i - x_j||², e_ij)

Equivariant coordinate update:
    x_i'   = x_i + C Σ_{j≠i} (x_i - x_j) φ_x(m_ij)

Message aggregation:
    agg_i  = Σ_j φ_a(m_ij)

Invariant node feature update:
    h_i'   = φ_h(h_i, agg_i)

where:
    x_i ∈ R^COORD_DIM  = equivariant coordinate (moves with rotations)
    h_i ∈ R^HIDDEN_DIM = invariant hidden state (unchanged by rotations)
    e_ij               = edge weight (|correlation|)
    φ_e, φ_x, φ_h     = MLPs with SiLU activations
```

The final score readout uses only h_i (invariant) → ETF scores are invariant.

---

## Architecture

```
Raw node features (in_dim)
         ↓
    Linear projection → h_i ∈ R^HIDDEN_DIM (invariant)
    Linear projection → x_i ∈ R^COORD_DIM  (equivariant coordinates)
         ↓
    N_EGNN_LAYERS × EGNNLayer
    (equivariant message passing over correlation graph)
         ↓
    Readout MLP(h_final) → scalar score per ETF
```

---

## Node Features

Each ETF node carries:
- Lagged returns: r_{t-1}, r_{t-5}, r_{t-21}, r_{t-63}
- Rolling 21d volatility (annualised)
- Rolling 63d momentum (annualised)
- Rolling correlation with VIX (macro sensitivity)
- Rolling correlation with DXY (macro sensitivity)

The first `COORD_DIM` features become equivariant coordinates; all features
are also used as invariant hidden state input.

---

## Graph Construction

Rolling `GRAPH_WINDOW=63d` Pearson correlation matrix.
Edge created where |ρ| > `CORR_THRESHOLD=0.20`.
Edge attribute = |ρ| (correlation strength).

---

## How It Differs From All 159 Existing Engines

| Property | CLIFFORD-NET | Standard GCN/GAT | **EGNN** |
|---|---|---|---|
| Equivariance type | Clifford grade-level | None | **E(n): rotations + translations** |
| Coordinate updates | Grade multiplication | None | **Equivariant coordinate shift** |
| Score invariance | Partial (grade-0) | No | **Provably invariant** |
| Geometric structure | Clifford algebra | Ignored | **Explicit x_i ∈ R^d positions** |
| Distance invariance | Via blade norms | No | **||x_i − x_j||² always invariant** |

---

## Two Workflows

### 1. `train.yml` — Manual, run weekly

```
GitHub → Actions → "EGNN Training (Manual)" → Run workflow
```

Trains EGNN on 2008 → TRAIN_END data. Saves checkpoint to `P2SAMAPA/p2-etf-egnn-model`.
**Run this first.**

### 2. `daily_run.yml` — Automated, Mon-Fri 21:45 UTC

Loads checkpoint, runs equivariant inference, pushes results to `P2SAMAPA/p2-etf-egnn-results`.

---

## Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `COORD_DIM` | 8 | Equivariant coordinate dimension |
| `HIDDEN_DIM` | 64 | Invariant node hidden dimension |
| `N_EGNN_LAYERS` | 4 | Message passing layers |
| `MLP_HIDDEN` | 128 | Hidden dim in φ_e, φ_x, φ_h MLPs |
| `GRAPH_WINDOW` | 63 | Rolling correlation window |
| `CORR_THRESHOLD` | 0.20 | Min correlation for edge creation |
| `N_EPOCHS` | 100 | Max training epochs |
| `LEARNING_RATE` | 1e-3 | Adam learning rate |

---

## Universes

| Universe | Tickers |
|---|---|
| EQUITY_SECTORS | SPY QQQ XLK XLF XLE XLV XLI XLY XLP XLU GDX XME IWF XSD XBI IWM |
| FI_COMMODITIES | TLT VCIT LQD HYG VNQ GLD SLV |
| COMBINED | All above |

---

## Output Files (per universe)

| File | Content |
|---|---|
| `egnn_YYYY-MM-DD_{slug}.json` | Latest scores, rankings, config |
| `daily_{slug}.csv` | Top pick, score, CASH flag, n_edges |
| `scores_{slug}.csv` | Full score history |
| `coords_{slug}.csv` | Equivariant coordinate history (x0,x1,x2 per ETF) |
| `rankings_{slug}.csv` | Full rank history |

---

## Streamlit Dashboard — 5 Tabs

1. **Rankings & Scores** — bar chart + score heatmap, top-N cards
2. **Coordinate Space** — 3D scatter of ETF positions in equivariant space, pairwise distance matrix
3. **Score History** — time-series + top-pick frequency
4. **Graph Structure** — edge count over time, equivariance explanation
5. **Full Table** — all scores + checkpoint info + daily summary

---

## References

- Satorras, V.G., Hoogeboom, E. & Welling, M. (2021). *E(n) Equivariant Graph Neural Networks.* ICML.
- Bronstein, M.M. et al. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.* arXiv.
- Thomas, N. et al. (2018). *Tensor Field Networks: Rotation- and Translation-Equivariant Neural Networks for 3D Point Clouds.* arXiv.
- Kipf, T. & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR.

---

*P2Quant Engine Suite · Built by P2SAMAPA*
