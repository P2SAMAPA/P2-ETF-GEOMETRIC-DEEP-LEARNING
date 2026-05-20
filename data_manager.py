"""data_manager.py — Data loading and feature engineering for EGNN engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

import config

ALL_TICKERS = sorted(set(
    config.EQUITY_SECTORS_TICKERS + config.FI_COMMODITIES_TICKERS
))


def load_data(token: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download master_data.parquet → (log_returns, macro_df)."""
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=token,
        cache_dir="./hf_cache",
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={"index": "Date"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True).set_index("Date")

    available   = [t for t in ALL_TICKERS if t in df.columns]
    prices      = df[available].ffill()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    macro_cols = [c for c in config.MACRO_COLS if c in df.columns]
    macro_df   = df[macro_cols].reindex(log_returns.index).ffill().fillna(0.0)

    print(
        f"Loaded {len(log_returns)} rows × {len(log_returns.columns)} ETFs"
        f" | Macro: {macro_cols}"
    )
    return log_returns, macro_df


def build_node_features(
    log_returns:  pd.DataFrame,
    macro_df:     pd.DataFrame,
    tickers:      list[str],
    t:            int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build node coordinates x_i and hidden features h_i for all ETFs at time t.

    Coordinate vector x_i ∈ R^COORD_DIM:
        Encodes the ETF's position in feature space. This is what
        EGNN treats geometrically — distances ||x_i - x_j||² encode
        how different two ETFs are in return-space.
        Components: [ret_1d, ret_5d, ret_21d, ret_63d,
                     rolling_vol_21, rolling_mom_63,
                     macro_vix_sensitivity, macro_dxy_sensitivity]

    Hidden vector h_i ∈ R^n_hidden_input:
        Invariant node attributes fed into the EGNN.
        Components: all coordinate features + additional context

    Returns
    -------
    coords  : (n_etf, COORD_DIM)  — geometric coordinates
    hidden  : (n_etf, n_features) — invariant node features
    """
    avail = [t_ for t_ in tickers if t_ in log_returns.columns]
    n_etf = len(avail)
    ret   = log_returns[avail].values    # (T, n_etf)
    mac   = macro_df.values              # (T, n_macro)

    coords_list  = []
    hidden_list  = []

    for i in range(n_etf):
        r = ret[:, i]
        feats = []

        # Lagged returns (coordinate + hidden)
        for lag in config.LOOKBACK_LAGS:
            idx = t - lag
            feats.append(float(r[idx]) if idx >= 0 else 0.0)

        # Rolling volatility (annualised)
        win_v = r[max(0, t - config.ROLLING_VOL_WINDOW): t]
        feats.append(float(np.std(win_v) * np.sqrt(252)) if len(win_v) > 1 else 0.0)

        # Rolling momentum (annualised)
        win_m = r[max(0, t - config.ROLLING_MOM_WINDOW): t]
        feats.append(float(np.mean(win_m) * 252) if len(win_m) > 1 else 0.0)

        # Macro sensitivities (rolling correlation of ETF return with each macro)
        for k in range(min(2, mac.shape[1])):   # VIX, DXY
            win_mac = mac[max(0, t - config.ROLLING_MOM_WINDOW): t, k]
            win_ret = r[max(0, t - config.ROLLING_MOM_WINDOW): t]
            min_len = min(len(win_mac), len(win_ret))
            if min_len > 5:
                corr = float(np.corrcoef(win_ret[:min_len], win_mac[:min_len])[0, 1])
                corr = 0.0 if np.isnan(corr) else corr
            else:
                corr = 0.0
            feats.append(corr)

        hidden_list.append(feats)

    hidden_arr = np.array(hidden_list, dtype=np.float32)   # (n_etf, n_feats)
    hidden_arr = np.clip(hidden_arr, -10.0, 10.0)

    # First COORD_DIM features become coordinates; rest stay as hidden-only
    n_coord = min(config.COORD_DIM, hidden_arr.shape[1])
    coords  = hidden_arr[:, :n_coord].copy()

    return coords, hidden_arr


def build_edge_index(
    log_returns: pd.DataFrame,
    tickers:     list[str],
    t:           int,
    window:      int = config.GRAPH_WINDOW,
    threshold:   float = config.CORR_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Build edge index and edge attributes from rolling correlation matrix.

    Returns
    -------
    edge_index : (2, n_edges) — pairs of connected node indices
    edge_attr  : (n_edges, 1) — edge weights (|correlation|)
    """
    avail = [t_ for t_ in tickers if t_ in log_returns.columns]
    n_etf = len(avail)
    win   = log_returns[avail].values[max(0, t - window): t]

    if win.shape[0] < 5:
        # No edges — return empty
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0, 1), dtype=np.float32)

    corr = np.corrcoef(win.T)             # (n_etf, n_etf)
    corr = np.nan_to_num(corr, nan=0.0)

    src_list, dst_list, attr_list = [], [], []
    for i in range(n_etf):
        for j in range(n_etf):
            if i != j and abs(corr[i, j]) >= threshold:
                src_list.append(i)
                dst_list.append(j)
                attr_list.append(abs(corr[i, j]))

    if not src_list:
        # Add self-loops as fallback
        src_list = list(range(n_etf))
        dst_list = list(range(n_etf))
        attr_list = [1.0] * n_etf

    edge_index = np.array([src_list, dst_list], dtype=np.int64)
    edge_attr  = np.array(attr_list, dtype=np.float32).reshape(-1, 1)
    return edge_index, edge_attr
