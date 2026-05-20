"""trainer_model.py — EGNN training script. Run via train.yml (manual, weekly)."""

from __future__ import annotations

import argparse
import io
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import HfApi

import config
import data_manager
from egnn import EGNNModel


def _listnet_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ListNet cross-sectional ranking loss. Directly optimises ETF ranking."""
    p_pred   = torch.softmax(pred   / 1.0,  dim=-1)
    p_target = torch.softmax(target / 0.01, dim=-1)
    return -(p_target * torch.log(p_pred + 1e-8)).sum()


def train(
    universe_name: str,
    tickers:       list[str],
    log_returns,
    macro_df,
    n_epochs:      int,
    device:        torch.device,
    token:         str,
) -> None:
    print(f"\n{'='*60}")
    print(f"EGNN Training — Universe: {universe_name}")
    print(f"Device: {device} | Epochs: {n_epochs}")
    print(f"Train data: 2008 → {config.TRAIN_END}")
    print(f"{'='*60}")

    avail  = [t for t in tickers if t in log_returns.columns]
    n_etf  = len(avail)
    mac_c  = [c for c in config.MACRO_COLS if c in macro_df.columns]
    dates  = log_returns.index
    train_end = np.searchsorted(dates, config.TRAIN_END, side="right")

    min_t = config.GRAPH_WINDOW + max(config.LOOKBACK_LAGS)

    # ── Build training dataset ────────────────────────────────────────────────
    print("Building training dataset...")
    train_steps = []
    for t in range(min_t, min(train_end, len(log_returns) - 1)):
        # Target: next-day cross-sectional returns
        target = torch.tensor(
            log_returns[avail].values[t + 1], dtype=torch.float32
        )
        train_steps.append(t)

    print(f"Training steps: {len(train_steps)}")

    # Determine feature dim from first sample
    coords0, hidden0 = data_manager.build_node_features(log_returns, macro_df, avail, min_t)
    in_dim = hidden0.shape[1]
    print(f"Node feature dim: {in_dim}  |  Coord dim: {config.COORD_DIM}  |  n_etf: {n_etf}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = EGNNModel(in_dim=in_dim).to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.LEARNING_RATE,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    history = {"train_loss": [], "val_loss": []}
    best_loss  = float("inf")
    patience_c = 0
    best_state = None

    # Val steps: last 20% of train period
    val_start = int(len(train_steps) * 0.80)
    tr_steps  = train_steps[:val_start]
    vl_steps  = train_steps[val_start:]

    rng = np.random.default_rng(42)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        model.train()
        epoch_losses = []
        rng.shuffle(tr_steps)

        for i in range(0, len(tr_steps), config.BATCH_SIZE):
            batch_t = tr_steps[i: i + config.BATCH_SIZE]
            batch_loss = torch.tensor(0.0, device=device)

            for t in batch_t:
                coords, hidden = data_manager.build_node_features(
                    log_returns, macro_df, avail, t)
                ei, ea = data_manager.build_edge_index(log_returns, avail, t)

                if ei.shape[1] == 0:
                    continue

                node_t = torch.tensor(hidden, dtype=torch.float32, device=device)
                ei_t   = torch.tensor(ei,     dtype=torch.long,    device=device)
                ea_t   = torch.tensor(ea,     dtype=torch.float32, device=device)
                tgt_t  = torch.tensor(
                    log_returns[avail].values[t + 1],
                    dtype=torch.float32, device=device)

                scores, _, _ = model(node_t, ei_t, ea_t)

                # Combined loss: MSE + ListNet ranking
                mse_l  = nn.functional.mse_loss(scores, tgt_t)
                rank_l = _listnet_loss(scores, tgt_t)
                batch_loss = batch_loss + mse_l + 0.30 * rank_l

            if batch_loss.requires_grad:
                optimizer.zero_grad()
                batch_loss = batch_loss / max(len(batch_t), 1)
                batch_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
                epoch_losses.append(batch_loss.item())

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for t in vl_steps[:100]:   # cap for speed
                coords, hidden = data_manager.build_node_features(
                    log_returns, macro_df, avail, t)
                ei, ea = data_manager.build_edge_index(log_returns, avail, t)
                if ei.shape[1] == 0:
                    continue
                node_t = torch.tensor(hidden, dtype=torch.float32, device=device)
                ei_t   = torch.tensor(ei,     dtype=torch.long,    device=device)
                ea_t   = torch.tensor(ea,     dtype=torch.float32, device=device)
                tgt_t  = torch.tensor(
                    log_returns[avail].values[t + 1],
                    dtype=torch.float32, device=device)
                scores, _, _ = model(node_t, ei_t, ea_t)
                val_losses.append(nn.functional.mse_loss(scores, tgt_t).item())

        tr_l  = float(np.mean(epoch_losses))  if epoch_losses else 0.0
        val_l = float(np.mean(val_losses))    if val_losses   else 0.0
        history["train_loss"].append(tr_l)
        history["val_loss"].append(val_l)

        elapsed = time.time() - t0
        if epoch % 10 == 0 or epoch == n_epochs:
            print(f"  Epoch {epoch:4d}/{n_epochs} | "
                  f"train={tr_l:.6f}  val={val_l:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  [{elapsed:.1f}s]")

        if val_l < best_loss - 1e-6:
            best_loss  = val_l
            patience_c = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_c += 1
            if patience_c >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── Save to HF ────────────────────────────────────────────────────────────
    slug = universe_name.lower().replace("_", "-")
    api  = HfApi(token=token)
    api.create_repo(repo_id=config.HF_MODEL_REPO, repo_type="model",
                    exist_ok=True, private=False)

    ckpt = {
        "model_state_dict": best_state,
        "in_dim":           in_dim,
        "n_etf":            n_etf,
        "tickers":          avail,
        "universe":         universe_name,
        "train_date":       config.TODAY,
        "best_val_loss":    best_loss,
        "config": {
            "coord_dim": config.COORD_DIM,
            "hidden_dim": config.HIDDEN_DIM,
            "n_layers":  config.N_EGNN_LAYERS,
            "mlp_hidden":config.MLP_HIDDEN,
            "train_end": config.TRAIN_END,
        },
    }
    buf = io.BytesIO()
    torch.save(ckpt, buf); buf.seek(0)

    api.upload_file(path_or_fileobj=buf,
                    path_in_repo=config.CKPT_MODEL.format(slug=slug),
                    repo_id=config.HF_MODEL_REPO, repo_type="model",
                    commit_message=f"EGNN model {slug} — {config.TODAY}")

    meta = {"universe": universe_name, "train_date": config.TODAY,
            "best_val_loss": best_loss, "in_dim": in_dim,
            "tickers": avail, "history": history}
    api.upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(meta, indent=2).encode()),
        path_in_repo=config.CKPT_META.format(slug=slug),
        repo_id=config.HF_MODEL_REPO, repo_type="model",
        commit_message=f"EGNN meta {slug} — {config.TODAY}")

    print(f"  ✅ Model → {config.HF_MODEL_REPO}/{config.CKPT_MODEL.format(slug=slug)}")
    print(f"  Best val loss: {best_loss:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", default="ALL")
    parser.add_argument("--epochs",   type=int, default=config.N_EPOCHS)
    args = parser.parse_args()

    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    log_returns, macro_df = data_manager.load_data(token=token)

    target = args.universe.upper()
    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target:
            continue
        train(universe_name=universe_name, tickers=tickers,
              log_returns=log_returns, macro_df=macro_df,
              n_epochs=args.epochs, device=device, token=token)

    print("\n✅ EGNN training complete.")


if __name__ == "__main__":
    main()
