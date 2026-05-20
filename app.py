"""app.py — Geometric Deep Learning (EGNN) Engine · Streamlit Dashboard."""
from __future__ import annotations
import os
from io import StringIO
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="EGNN Geometric DL · P2Quant", layout="wide", page_icon="🔷")

HF_TOKEN = os.environ.get("HF_TOKEN")
BASE_RAW = f"https://huggingface.co/datasets/{config.HF_OUTPUT_REPO}/resolve/main"
BASE_API = f"https://huggingface.co/api/datasets/{config.HF_OUTPUT_REPO}/tree/main"
HEADERS  = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
PALETTE  = ["#1B4F8A","#27AE60","#E74C3C","#F39C12","#8E44AD","#148F77",
            "#CA6F1E","#2471A3","#CB4335","#1A5276","#117A65","#B7950B",
            "#884EA0","#1F618D","#B9770E","#922B21"]

def sc(v):
    if v>=0.5: return "#1D9E75"
    if v>=0.0: return "#82C3A9"
    if v>=-0.5: return "#F0A07A"
    return "#E74C3C"

def fmt(v,d=4): return f"{v:+.{d}f}"

@st.cache_data(ttl=3600, show_spinner="Loading EGNN results…")
def load_json(universe):
    slug=universe.lower().replace("_","-")
    try:
        r=requests.get(BASE_API,headers=HEADERS,timeout=30)
        if r.status_code!=200: return None
        files=sorted(f["path"] for f in r.json() if f["path"].endswith(".json"))
        matches=[f for f in files if f"_{slug}.json" in f]
        if not matches: return None
        resp=requests.get(f"{BASE_RAW}/{matches[-1]}",headers=HEADERS,timeout=30)
        resp.raise_for_status(); return resp.json()
    except Exception: return None

@st.cache_data(ttl=3600, show_spinner="Loading history…")
def load_csv(filename):
    try:
        r=requests.get(f"{BASE_RAW}/{filename}",headers=HEADERS,timeout=60)
        if r.status_code!=200: return None
        df=pd.read_csv(StringIO(r.text),index_col=0,parse_dates=True)
        return df if not df.empty else None
    except Exception: return None

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    universe=st.selectbox("Universe",list(config.UNIVERSES.keys()))
    st.divider()
    st.markdown(f"**Architecture:** EGNN ({config.N_EGNN_LAYERS} layers)")
    st.markdown(f"**Coord dim:** {config.COORD_DIM}  |  **Hidden dim:** {config.HIDDEN_DIM}")
    st.markdown(f"**Graph window:** {config.GRAPH_WINDOW}d")
    st.markdown(f"**Corr threshold:** {config.CORR_THRESHOLD}")
    st.markdown(f"**OOS from:** {config.OOS_START}")
    st.markdown(f"**Next trading day:** {next_trading_day()}")
    st.divider()
    st.markdown("**Equivariance:**")
    st.markdown("- Scores **invariant** to rotations/translations of feature space")
    st.markdown("- Coordinates **equivariant** — transform consistently with inputs")
    st.markdown("- Distances ||x_i − x_j||² always invariant → stable edges")
    st.divider()
    st.markdown("**Workflows:**")
    st.markdown("🔷 `train.yml` — weekly manual training")
    st.markdown("📅 `daily_run.yml` — daily auto inference")
    if st.button("🔄 Refresh"):
        st.cache_data.clear(); st.rerun()

st.markdown("# 🔷 Geometric Deep Learning Engine — E(n)-Equivariant GNN")
st.caption(
    "Each ETF = point in R^d coordinate space · "
    "E(n)-equivariant message passing (EGNN, Satorras et al. 2021) · "
    "Scores invariant to rotations/translations of feature space · "
    "Strictly more principled than standard GNNs for multi-scale financial features"
)

slug      = universe.lower().replace("_","-")
data      = load_json(universe)
daily_df  = load_csv(f"daily_{slug}.csv")
score_df  = load_csv(f"scores_{slug}.csv")
coord_df  = load_csv(f"coords_{slug}.csv")
rank_df   = load_csv(f"rankings_{slug}.csv")

if data is None:
    st.warning("⚠️ No results found. Run `train.yml` first, then `daily_run.yml`.")
    st.stop()

latest_scores = data.get("latest_scores", {})
latest_ranked = data.get("latest_ranked", [])
latest_date   = data.get("latest_date", "?")
run_date      = data.get("run_date", "?")
ckpt_meta     = data.get("ckpt_meta", {})
cfg           = data.get("config", {})

k1,k2,k3,k4 = st.columns(4)
k1.metric("Run Date", run_date)
k2.metric("Latest Date", latest_date)
k3.metric("Model Trained", ckpt_meta.get("train_date","?"))
k4.metric("Val Loss", f"{ckpt_meta.get('best_val_loss',0):.6f}" if ckpt_meta.get("best_val_loss") else "?")

if latest_ranked:
    top = latest_ranked[0]
    cash= top.get("composite_score",0) < config.CASH_THRESHOLD
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("🏆 Top Pick", "CASH" if cash else top["ticker"])
    m2.metric("Top Score", fmt(top.get("composite_score",0)))
    m3.metric("Universe", universe)
    m4.metric("CASH Signal", "Yes ⚠️" if cash else "No ✅")

st.divider()

tab1,tab2,tab3,tab4,tab5 = st.tabs([
    "🎯 Rankings & Scores",
    "🔷 Coordinate Space",
    "📈 Score History",
    "🕸️ Graph Structure",
    "📋 Full Table",
])

with tab1:
    st.subheader(f"EGNN Rankings as of {latest_date}")
    tickers_r=[r["ticker"] for r in latest_ranked]
    scores_r =[r.get("composite_score",0) for r in latest_ranked]
    col_l,col_r = st.columns(2)
    with col_l:
        fig=go.Figure(go.Bar(y=tickers_r,x=scores_r,orientation="h",
            marker_color=[sc(s) for s in scores_r],
            text=[fmt(s) for s in scores_r],textposition="outside"))
        fig.add_vline(x=0,line_dash="dot",line_color="gray")
        fig.update_layout(title="E(n)-equivariant GNN score (rotation-invariant)",
            xaxis_title="Composite z-score",yaxis=dict(autorange="reversed"),
            height=max(300,len(tickers_r)*30),margin=dict(t=50,b=20,l=60,r=80),
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True,key="rank_bar")
    with col_r:
        if score_df is not None:
            etf_cols=[c for c in score_df.columns if c in config.UNIVERSES[universe]]
            recent=score_df[etf_cols].tail(252)
            fig2=go.Figure(go.Heatmap(z=recent.values.T,
                x=recent.index.strftime("%Y-%m-%d"),y=list(recent.columns),
                colorscale="RdYlGn",zmid=0,colorbar=dict(title="Score")))
            fig2.update_layout(title="Score heatmap — last 252 days",
                height=max(300,len(etf_cols)*22+80),
                margin=dict(t=50,b=60,l=60,r=20),
                xaxis=dict(tickangle=-45,nticks=10),
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2,use_container_width=True,key="heat_right")
    st.markdown(f"### 🎯 Top {config.TOP_N} for {next_trading_day()}")
    cols=st.columns(config.TOP_N)
    for i,row in enumerate(latest_ranked[:config.TOP_N]):
        with cols[i]:
            s=row.get("composite_score",0)
            st.markdown(
                f"**#{i+1} {row['ticker']}**\n\n"
                f"Score: `{fmt(s)}`\n\n"
                f"Rank: `{row.get('rank',i+1)}`\n\n"
                f'<span style="background:{sc(s)};color:white;padding:2px 8px;'
                f'border-radius:8px;font-size:11px">#{row.get("rank",i+1)}</span>',
                unsafe_allow_html=True)

with tab2:
    st.subheader("ETF Coordinate Space — Equivariant Geometry")
    st.caption(
        "Each ETF is a point in R^d equivariant coordinate space. "
        "Distances ||x_i − x_j||² between ETFs are invariant to rotations/translations. "
        "ETFs close in coordinate space = similar geometric structure = correlated behaviour. "
        "Coordinates evolve daily as the EGNN updates them through message passing."
    )
    if coord_df is not None and not coord_df.empty:
        # Get columns for x0, x1, x2 for each ETF
        tkrs_plot = [t for t in (ckpt_meta.get("tickers") or tickers_r)
                     if f"{t}_x0" in coord_df.columns][:12]
        if len(tkrs_plot) >= 2:
            last_row = coord_df.iloc[-1]
            xs = [float(last_row.get(f"{t}_x0", 0)) for t in tkrs_plot]
            ys = [float(last_row.get(f"{t}_x1", 0)) for t in tkrs_plot]
            zs = [float(last_row.get(f"{t}_x2", 0)) for t in tkrs_plot]
            score_vals = [latest_scores.get(t, {}).get("composite_score", 0) for t in tkrs_plot]
            fig_coord = go.Figure(go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers+text",
                text=tkrs_plot, textposition="top center",
                marker=dict(size=8, color=score_vals, colorscale="RdYlGn",
                            showscale=True, colorbar=dict(title="Score"))))
            fig_coord.update_layout(
                title=f"ETF positions in equivariant coordinate space — {latest_date}",
                scene=dict(xaxis_title="x₀",yaxis_title="x₁",zaxis_title="x₂"),
                height=500, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_coord, use_container_width=True, key="coord_3d")
            st.info(
                "**Key insight:** ETFs that cluster together in coordinate space have similar "
                "geometric relationships to their neighbours in the correlation graph. "
                "The EGNN moves these coordinates through message passing — ETFs shift "
                "toward ETFs they are strongly correlated with, and away from anticorrelated ETFs. "
                "The final position encodes the ETF's geometric role in the market structure."
            )
        # Pairwise distance matrix
        if len(tkrs_plot) >= 2:
            st.markdown("**Pairwise equivariant distances ||x_i − x_j||² (latest)**")
            pts = np.array(list(zip(xs,ys,zs)))
            dist_mat = np.sum((pts[:,None,:]-pts[None,:,:])**2, axis=-1)
            fig_dist = go.Figure(go.Heatmap(
                z=dist_mat, x=tkrs_plot, y=tkrs_plot,
                colorscale="Blues_r", colorbar=dict(title="Sq. Distance")))
            fig_dist.update_layout(
                title="Equivariant pairwise distance matrix (dark = close in coord space)",
                height=max(300,len(tkrs_plot)*28+80),
                margin=dict(t=40,b=60,l=60,r=20),
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_dist,use_container_width=True,key="dist_mat")
    else:
        st.info("Coordinate history not available.")

with tab3:
    st.subheader("Score History")
    if score_df is not None:
        etf_cols=[c for c in score_df.columns if c in config.UNIVERSES[universe]]
        sel=st.multiselect("Select ETFs",etf_cols,default=etf_cols[:6],key="score_sel")
        period=st.radio("Period",["Last 2 years","Last 5 years","Full OOS"],
                        horizontal=True,key="score_period")
        df_s=score_df.copy()
        if period=="Last 2 years": df_s=df_s[df_s.index>="2024-01-01"]
        elif period=="Last 5 years": df_s=df_s[df_s.index>="2021-01-01"]
        if sel:
            fig_s=go.Figure()
            for i,tkr in enumerate(sel):
                if tkr in df_s.columns:
                    fig_s.add_trace(go.Scatter(x=df_s.index,y=df_s[tkr],
                        mode="lines",name=tkr,
                        line=dict(width=1.4,color=PALETTE[i%len(PALETTE)])))
            fig_s.add_hline(y=0,line_dash="dot",line_color="gray")
            fig_s.update_layout(title="EGNN composite score (z-scored)",
                yaxis_title="Score",height=400,
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h",yanchor="bottom",y=1.02))
            st.plotly_chart(fig_s,use_container_width=True,key="score_ts")
        if daily_df is not None and "top_ticker" in daily_df.columns:
            picks=daily_df["top_ticker"].value_counts()
            fig_f=go.Figure(go.Bar(x=picks.index,y=picks.values,
                marker_color="#1B4F8A",text=picks.values,textposition="outside"))
            fig_f.update_layout(title="Top-pick frequency (OOS)",
                yaxis_title="Days as #1 EGNN pick",height=280,
                plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_f,use_container_width=True,key="pick_freq")
    else:
        st.info("No score history found.")

with tab4:
    st.subheader("Graph Structure")
    st.caption(
        f"Rolling {config.GRAPH_WINDOW}d correlation graph. "
        f"Edges created where |ρ| > {config.CORR_THRESHOLD}. "
        "Edge weights = |correlation|. EGNN message passing operates on this graph."
    )
    if daily_df is not None and "n_edges" in daily_df.columns:
        fig_e=go.Figure(go.Scatter(x=daily_df.index,y=daily_df["n_edges"],
            mode="lines",line=dict(color="#1B4F8A",width=1.5),
            fill="tozeroy",fillcolor="rgba(27,79,138,0.08)"))
        fig_e.update_layout(title="Number of graph edges over time",
            yaxis_title="Edges",height=280,
            plot_bgcolor="rgba(0,0,0,0)",paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_e,use_container_width=True,key="edge_ts")
    st.markdown("**Why equivariance matters for this graph:**")
    st.markdown(
        "Standard GNNs (GCN, GAT) treat node feature vectors as plain tensors. "
        "If you rescale or rotate your features (e.g. VIX in points vs z-score), "
        "the output changes arbitrarily. **EGNN's outputs are provably invariant** — "
        "any rotation or translation of the coordinate vectors x_i produces "
        "exactly the same invariant scores. This means the ranking is stable "
        "across different feature normalisations, scales, and units."
    )

with tab5:
    st.subheader(f"Full EGNN Table — {latest_date}")
    if latest_ranked:
        rows=[{"Rank":i+1,"Ticker":r["ticker"],
               "Composite Score":fmt(r.get("composite_score",0))}
              for i,r in enumerate(latest_ranked)]
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True,height=600)
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        st.markdown("**Checkpoint Info**"); st.json(ckpt_meta)
    with c2:
        st.markdown("**Engine Configuration**"); st.json(cfg)
    if daily_df is not None:
        st.divider()
        st.markdown("**Daily summary (last 20 days)**")
        st.dataframe(daily_df.tail(20),use_container_width=True)
    st.divider()
    st.caption(
        f"P2Quant EGNN Engine · Run: {run_date} · "
        f"E(n)-Equivariant GNN (Satorras et al. 2021) · "
        f"{config.N_EGNN_LAYERS} EGNN layers · coord_dim={config.COORD_DIM} · "
        f"Data: {config.HF_DATA_REPO}"
    )
