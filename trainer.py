"""trainer.py — EGNN daily inference orchestrator with HF push."""
from __future__ import annotations
import io, json, os
import torch
from huggingface_hub import HfApi
import config, data_manager
from engine import run_engine

def push_results(result: dict, universe: str, token: str) -> None:
    slug = universe.lower().replace("_","-")
    api  = HfApi(token=token)
    api.create_repo(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset",
                    exist_ok=True, private=False)
    ckpt = result.get("ckpt_meta", {})
    output = {
        "run_date": config.TODAY, "universe": universe,
        "latest_date": result["latest_date"],
        "latest_scores": result["latest_scores"],
        "latest_ranked": [{"ticker": t, **v} for t, v in result["latest_ranked"]],
        "ckpt_meta": {"train_date": ckpt.get("train_date","?"),
                      "best_val_loss": ckpt.get("best_val_loss", None)},
        "config": {"coord_dim": config.COORD_DIM, "hidden_dim": config.HIDDEN_DIM,
                   "n_layers": config.N_EGNN_LAYERS, "graph_window": config.GRAPH_WINDOW,
                   "corr_threshold": config.CORR_THRESHOLD,
                   "cash_threshold": config.CASH_THRESHOLD,
                   "top_n": config.TOP_N, "oos_start": config.OOS_START},
    }
    def _push(data: bytes, path: str, msg: str):
        api.upload_file(path_or_fileobj=io.BytesIO(data), path_in_repo=path,
                        repo_id=config.HF_OUTPUT_REPO, repo_type="dataset",
                        commit_message=msg)
    _push(json.dumps(output, indent=2, default=str).encode(),
          f"egnn_{config.TODAY}_{slug}.json", f"EGNN results {config.TODAY} — {slug}")
    for name, df in [("daily", result["daily_df"]), ("scores", result["score_df"]),
                     ("coords", result["coord_df"]), ("rankings", result["ranking_df"])]:
        _push(df.to_csv().encode(), f"{name}_{slug}.csv",
              f"{name} {config.TODAY} — {slug}")
    print(f"  ✅ Pushed → {config.HF_OUTPUT_REPO}/egnn_{config.TODAY}_{slug}.json")

def main() -> None:
    token = config.HF_TOKEN
    if not token:
        print("HF_TOKEN not set — aborting."); return
    target = os.environ.get("EGNN_UNIVERSE", "ALL").upper()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_returns, macro_df = data_manager.load_data(token=token)
    for universe_name, tickers in config.UNIVERSES.items():
        if target != "ALL" and universe_name != target: continue
        result = run_engine(log_returns=log_returns, macro_df=macro_df,
                            universe_tickers=tickers, universe_name=universe_name,
                            token=token, device=device)
        push_results(result, universe_name, token)
    print("\n✅ EGNN daily inference complete.")

if __name__ == "__main__":
    main()
