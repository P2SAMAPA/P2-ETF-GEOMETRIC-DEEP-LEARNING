"""engine.py — EGNN daily walk-forward inference engine."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

import config
import data_manager
from egnn import EGNNModel


def _load_checkpoint(universe: str, token: str | None, device: torch.device) -> dict:
    slug = universe.lower().replace("_", "-")
    f = hf_hub_download(repo_id=config.HF_MODEL_REPO,
                        filename=config.CKPT_MODEL.format(slug=slug),
                        repo_type="model", token=token, cache_dir="./hf_cache")
    ckpt = torch.load(io.BytesIO(open(f, "rb").read()),
                      map_location=device, weights_only=False)
    model = EGNNModel(in_dim=ckpt["in_dim"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"  Loaded EGNN: in_dim={ckpt['in_dim']}  "
          f"n_etf={ckpt['n_etf']}  trained={ckpt.get('train_date','?')}")
    return {"model": model, "ckpt": ckpt, "tickers": ckpt["tickers"]}


def zscore_cross(arr: np.ndarray) -> np.ndarray:
    mu = arr.mean(); std = arr.std() + 1e-8
    return (arr - mu) / std


def run_engine(log_returns, macro_df, universe_tickers, universe_name,
               token=None, device=None) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    avail = [t for t in universe_tickers if t in log_returns.columns]
    print(f"\n{'='*60}\nUniverse: {universe_name}  ({len(avail)} ETFs)\n{'='*60}")

    ckpt_data = _load_checkpoint(universe_name, token, device)
    model     = ckpt_data["model"]
    tickers   = ckpt_data["tickers"]

    oos_start = pd.Timestamp(config.OOS_START)
    dates     = log_returns.index
    min_t     = config.GRAPH_WINDOW + max(config.LOOKBACK_LAGS)

    score_records, coord_records, ranking_records, daily_records = [], [], [], []
    n_scored = 0

    for t in range(min_t, len(log_returns)):
        date = dates[t]
        if date < oos_start:
            continue

        coords, hidden = data_manager.build_node_features(log_returns, macro_df, tickers, t)
        ei, ea = data_manager.build_edge_index(log_returns, tickers, t)

        node_t = torch.tensor(hidden, dtype=torch.float32, device=device)
        ei_t   = torch.tensor(ei,     dtype=torch.long,    device=device)
        ea_t   = torch.tensor(ea,     dtype=torch.float32, device=device)

        with torch.no_grad():
            raw_scores, _, x_out = model(node_t, ei_t, ea_t)

        scores_np  = raw_scores.cpu().numpy()
        x_np       = x_out.cpu().numpy()
        composite  = zscore_cross(scores_np)
        ranked_idx = np.argsort(composite)[::-1]
        top_ticker = tickers[ranked_idx[0]]
        top_score  = float(composite[ranked_idx[0]])
        cash_flag  = top_score < config.CASH_THRESHOLD

        ds = date.strftime("%Y-%m-%d")
        n_scored += 1

        score_records.append({"date": ds,
            **{tickers[i]: round(float(composite[i]), 6) for i in range(len(tickers))}})
        coord_records.append({"date": ds,
            **{f"{tickers[i]}_x{k}": round(float(x_np[i, k]), 6)
               for i in range(len(tickers)) for k in range(min(3, x_np.shape[1]))}})
        ranking_records.append({"date": ds,
            **{tickers[ranked_idx[r]]: r + 1 for r in range(len(tickers))}})
        daily_records.append({
            "date": ds, "top_ticker": "CASH" if cash_flag else top_ticker,
            "top_score": round(top_score, 6), "cash_flag": cash_flag,
            "n_edges": ei.shape[1],
        })

        if n_scored % 252 == 0 or t == len(log_returns) - 1:
            top5 = [(tickers[ranked_idx[r]], round(float(composite[ranked_idx[r]]), 3))
                    for r in range(min(5, len(tickers)))]
            print(f"  {ds} | " + "  ".join(f"{tk}({sc:+.2f})" for tk, sc in top5)
                  + (" [CASH]" if cash_flag else ""))

    latest_score   = score_records[-1]
    latest_ranking = ranking_records[-1]
    latest_date    = daily_records[-1]["date"]

    latest_out = {}
    for tkr in tickers:
        latest_out[tkr] = {
            "composite_score": latest_score[tkr],
            "rank": int(latest_ranking[tkr]),
        }
    latest_ranked = sorted(latest_out.items(),
                           key=lambda x: x[1]["composite_score"], reverse=True)

    print(f"\n  Latest ({latest_date}) top-{config.TOP_N}: "
          + "  ".join(f"{t}({v['composite_score']:+.3f})"
                      for t, v in latest_ranked[:config.TOP_N]))
    print(f"  Days scored (OOS): {n_scored}")

    return {
        "latest_date": latest_date, "latest_scores": latest_out,
        "latest_ranked": latest_ranked,
        "daily_df":   pd.DataFrame(daily_records).set_index("date"),
        "score_df":   pd.DataFrame(score_records).set_index("date"),
        "coord_df":   pd.DataFrame(coord_records).set_index("date"),
        "ranking_df": pd.DataFrame(ranking_records).set_index("date"),
        "universe": universe_name, "n_etf": len(tickers),
        "ckpt_meta": ckpt_data["ckpt"],
    }
