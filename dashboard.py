#!/usr/bin/env python3
# Crystal Battle training dashboard
# Usage: streamlit run dashboard.py

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

PROJECT = Path(__file__).parent
MASTER_CSV = PROJECT / "eval_history.csv"
LIVE_CSV = PROJECT / "live_metrics.csv"
PBT_CSV = PROJECT / "pbt_history.csv"

# ============================================================
# HISTORICAL DATA (pre-CSV runs)
# ============================================================
HISTORICAL = pd.DataFrame([
    # Run 10 (flat MLP, opp move prediction)
    {"run": "run10", "steps": 500_000, "vs_random": 0.880, "vs_maxdmg": 0.400},
    {"run": "run10", "steps": 1_000_000, "vs_random": 0.835, "vs_maxdmg": 0.430},
    {"run": "run10", "steps": 1_500_000, "vs_random": 0.855, "vs_maxdmg": 0.460},
    {"run": "run10", "steps": 2_000_000, "vs_random": 0.845, "vs_maxdmg": 0.475},
    {"run": "run10", "steps": 2_500_000, "vs_random": 0.870, "vs_maxdmg": 0.490},
    {"run": "run10", "steps": 3_000_000, "vs_random": 0.835, "vs_maxdmg": 0.475},
    {"run": "run10", "steps": 3_500_000, "vs_random": 0.850, "vs_maxdmg": 0.440},
    {"run": "run10", "steps": 4_000_000, "vs_random": 0.875, "vs_maxdmg": 0.480},
    {"run": "run10", "steps": 4_500_000, "vs_random": 0.870, "vs_maxdmg": 0.480},
    {"run": "run10", "steps": 5_000_000, "vs_random": 0.875, "vs_maxdmg": 0.475},
    {"run": "run10", "steps": 5_500_000, "vs_random": 0.860, "vs_maxdmg": 0.465},
    # Run 13 (attention v1)
    {"run": "run13", "steps": 500_000, "vs_random": 0.885, "vs_maxdmg": 0.495},
    {"run": "run13", "steps": 1_000_000, "vs_random": 0.895, "vs_maxdmg": 0.465},
    {"run": "run13", "steps": 1_500_000, "vs_random": 0.910, "vs_maxdmg": 0.490},
    {"run": "run13", "steps": 2_000_000, "vs_random": 0.895, "vs_maxdmg": 0.480},
    {"run": "run13", "steps": 2_500_000, "vs_random": 0.845, "vs_maxdmg": 0.515},
    {"run": "run13", "steps": 3_000_000, "vs_random": 0.900, "vs_maxdmg": 0.525},
    {"run": "run13", "steps": 3_500_000, "vs_random": 0.860, "vs_maxdmg": 0.510},
    {"run": "run13", "steps": 4_000_000, "vs_random": 0.920, "vs_maxdmg": 0.495},
])

RUN_META = {
    "run10": "R10: flat MLP + opp prediction",
    "run11": "R11: more obs features",
    "run12": "R12: blended reward",
    "run13": "R13: attention v1 *",
    "run14": "R14: attention v3",
    "run15": "R15: attention v2",
    "run16": "R16: low LR (1e-4)",
    "run16b": "R16b: smart opponents",
    "run17": "R17: v1 + steep LR decay",
    "run18": "R18: OU tier constraint",
    "run19": "R19: self-play bolt-on",
    "run20": "R20: 3-phase curriculum",
    "run21": "R21: enriched obs (256d)",
}


def load_eval_data() -> pd.DataFrame:
    frames = [HISTORICAL]
    if MASTER_CSV.exists():
        csv = pd.read_csv(MASTER_CSV)
        frames.append(csv)
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["run", "steps"], keep="last")
    df = df.sort_values(["run", "steps"])
    return df


def load_live_data() -> pd.DataFrame | None:
    if not LIVE_CSV.exists():
        return None
    try:
        df = pd.read_csv(LIVE_CSV)
        if df.empty:
            return None
        # entropy is logged as negative, take abs
        if "entropy" in df.columns:
            df["entropy"] = df["entropy"].abs()
        return df
    except Exception:
        return None


# ============================================================
# EVAL CHARTS
# ============================================================
def make_win_rate_chart(df: pd.DataFrame) -> go.Figure:
    has_smart = "vs_smart" in df.columns and df["vs_smart"].notna().any()
    has_crystal = "vs_crystal" in df.columns and df["vs_crystal"].notna().any()
    n_rows = 2 + int(has_smart) + int(has_crystal)
    titles = ["vs MaxDamage", "vs Random"]
    if has_smart:
        titles.insert(1, "vs Smart")
    if has_crystal:
        titles.insert(2 if not has_smart else 2, "vs Crystal AI")

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=titles,
    )

    for run in df["run"].unique():
        rd = df[df["run"] == run]
        label = RUN_META.get(run, run)
        row_idx = 1
        fig.add_trace(go.Scatter(
            x=rd["steps"], y=rd["vs_maxdmg"] * 100,
            mode="lines+markers", name=label,
            marker=dict(size=5),
            legendgroup=run, showlegend=True,
        ), row=row_idx, col=1)
        row_idx += 1
        if has_smart:
            smart_data = rd[rd["vs_smart"].notna()]
            if len(smart_data) > 0:
                fig.add_trace(go.Scatter(
                    x=smart_data["steps"], y=smart_data["vs_smart"] * 100,
                    mode="lines+markers", name=label,
                    marker=dict(size=5),
                    legendgroup=run, showlegend=False,
                ), row=row_idx, col=1)
            row_idx += 1
        if has_crystal:
            crystal_data = rd[rd["vs_crystal"].notna()]
            if len(crystal_data) > 0:
                fig.add_trace(go.Scatter(
                    x=crystal_data["steps"], y=crystal_data["vs_crystal"] * 100,
                    mode="lines+markers", name=label,
                    marker=dict(size=5),
                    legendgroup=run, showlegend=False,
                ), row=row_idx, col=1)
            row_idx += 1
        fig.add_trace(go.Scatter(
            x=rd["steps"], y=rd["vs_random"] * 100,
            mode="lines+markers", name=label,
            marker=dict(size=5),
            legendgroup=run, showlegend=False,
        ), row=row_idx, col=1)

    fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=1, col=1)
    if has_smart:
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
    for r in range(1, n_rows + 1):
        fig.update_yaxes(title_text="Win %", row=r, col=1)
    fig.update_xaxes(title_text="Training Steps", row=n_rows, col=1)
    fig.update_layout(height=250 * n_rows, legend=dict(orientation="h", y=-0.1))
    return fig


# ============================================================
# LIVE TRAINING CHARTS
# ============================================================
def make_live_training_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Explained Variance", "Entropy",
            "KL Divergence", "Clip Fraction",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    fig.add_trace(go.Scatter(
        x=df["steps"], y=df["exvar"],
        mode="lines", name="exvar",
        line=dict(color="#636EFA"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["steps"], y=df["entropy"],
        mode="lines", name="entropy",
        line=dict(color="#EF553B"),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=df["steps"], y=df["kl"],
        mode="lines", name="KL",
        line=dict(color="#00CC96"),
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["steps"], y=df["clip_frac"],
        mode="lines", name="clip",
        line=dict(color="#AB63FA"),
    ), row=2, col=2)

    fig.update_layout(height=450, showlegend=False)
    return fig


def make_live_loss_chart(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Total Loss", "Value Loss", "Policy Gradient Loss"),
    )

    fig.add_trace(go.Scatter(
        x=df["steps"], y=df["loss"],
        mode="lines", name="loss",
        line=dict(color="#636EFA"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["steps"], y=df["value_loss"],
        mode="lines", name="value_loss",
        line=dict(color="#EF553B"),
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=df["steps"], y=df["pg_loss"],
        mode="lines", name="pg_loss",
        line=dict(color="#00CC96"),
    ), row=1, col=3)

    fig.update_layout(height=300, showlegend=False)
    return fig


def make_live_actions_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    action_traces = [
        ("dmg_pct", "damage", "#EF553B"),
        ("setup_pct", "setup", "#AB63FA"),
        ("status_pct", "status", "#FFA15A"),
        ("other_pct", "other", "#B6E880"),
        ("switch_pct", "switch", "#636EFA"),
        ("fsw_pct", "forced_switch", "#19D3F3"),
    ]
    for col, name, color in action_traces:
        if col in df.columns and df[col].notna().any():
            fig.add_trace(go.Scatter(
                x=df["steps"], y=df[col] * 100,
                mode="lines", name=name,
                stackgroup="actions",
                line=dict(color=color),
            ))

    fig.update_layout(
        height=300,
        yaxis_title="Action %",
        xaxis_title="Training Steps",
        legend=dict(orientation="h", y=-0.25),
    )
    return fig


def make_live_reward_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["steps"], y=df["reward_mean"],
        mode="lines", name="mean",
        line=dict(color="#636EFA"),
    ))
    # shade +/- 1 std
    if "reward_std" in df.columns:
        upper = df["reward_mean"] + df["reward_std"]
        lower = df["reward_mean"] - df["reward_std"]
        fig.add_trace(go.Scatter(
            x=pd.concat([df["steps"], df["steps"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(99,110,250,0.15)",
            line=dict(width=0), name="+/- 1 std",
            showlegend=False,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        height=300,
        yaxis_title="Episode Reward",
        xaxis_title="Training Steps",
    )
    return fig


# ============================================================
# STREAMLIT LAYOUT
# ============================================================
st.set_page_config(page_title="Crystal Battle Dashboard", layout="wide")
st.title("Crystal Battle -- Training Dashboard")

# auto-refresh
refresh = st.sidebar.selectbox("Auto-refresh", ["Off", "10s", "30s", "60s", "5m"])
refresh_map = {"Off": None, "10s": 10, "30s": 30, "60s": 60, "5m": 300}
if refresh_map[refresh]:
    st.sidebar.caption(f"Refreshing every {refresh}")

eval_df = load_eval_data()
live_df = load_live_data()

# ---- Sidebar: run filter ----
all_runs = sorted(eval_df["run"].unique())
selected = st.sidebar.multiselect(
    "Runs to display",
    options=all_runs,
    default=all_runs,
    format_func=lambda r: RUN_META.get(r, r),
)
eval_df = eval_df[eval_df["run"].isin(selected)]

# ============================================================
# LIVE TRAINING (current run)
# ============================================================
if live_df is not None and not live_df.empty:
    run_id = live_df["run"].iloc[-1]
    latest = live_df.iloc[-1]
    st.header(f"Live Training: {RUN_META.get(run_id, run_id)}")

    # current stats
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Steps", f"{int(latest['steps']):,}")
    c2.metric("FPS", f"{latest['fps']:.0f}" if pd.notna(latest['fps']) else "-")
    c3.metric("LR", f"{latest['lr']:.2e}" if pd.notna(latest['lr']) else "-")
    c4.metric("Exvar", f"{latest['exvar']:.3f}" if pd.notna(latest['exvar']) else "-")
    c5.metric("Entropy", f"{latest['entropy']:.3f}" if pd.notna(latest['entropy']) else "-")
    c6.metric("Reward", f"{latest['reward_mean']:.3f}" if pd.notna(latest.get('reward_mean')) else "-")

    # training health
    st.subheader("Training Health")
    st.plotly_chart(make_live_training_chart(live_df), width="stretch")

    # losses
    st.subheader("Losses")
    st.plotly_chart(make_live_loss_chart(live_df), width="stretch")

    # actions + reward side by side
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Action Distribution")
        if live_df["dmg_pct"].notna().any():
            st.plotly_chart(make_live_actions_chart(live_df.dropna(subset=["dmg_pct"])),
                           width="stretch")
        else:
            st.info("No action data yet")
    with col_right:
        st.subheader("Episode Reward")
        if live_df["reward_mean"].notna().any():
            st.plotly_chart(make_live_reward_chart(live_df.dropna(subset=["reward_mean"])),
                           width="stretch")
        else:
            st.info("No reward data yet")

    st.divider()

# ============================================================
# EVAL HISTORY (all runs)
# ============================================================
st.header("Eval History")

# summary metrics
st.subheader("Best Results")
n_cols = min(len(selected), 5)
if n_cols > 0:
    cols = st.columns(n_cols)
    for i, run in enumerate(selected[:5]):
        rd = eval_df[eval_df["run"] == run]
        if rd.empty:
            continue
        best_md = rd["vs_maxdmg"].max() * 100
        peak_step = rd.loc[rd["vs_maxdmg"].idxmax(), "steps"]
        with cols[i % n_cols]:
            st.metric(
                label=RUN_META.get(run, run),
                value=f"{best_md:.1f}%",
                delta=f"peak @ {peak_step/1e6:.1f}M",
            )

# win rate chart
st.subheader("Win Rate Over Training")
st.plotly_chart(make_win_rate_chart(eval_df), width="stretch")

# action distribution from eval data
action_cols = ["dmg_pct", "status_pct", "switch_pct"]
has_actions = eval_df.dropna(subset=action_cols, how="all")
if not has_actions.empty:
    st.subheader("Action Distribution (Eval Checkpoints)")
    eval_action_traces = [
        ("dmg_pct", "damage", "solid", "#EF553B"),
        ("setup_pct", "setup", "dashdot", "#AB63FA"),
        ("status_pct", "status", "dot", "#FFA15A"),
        ("other_pct", "other", "dot", "#B6E880"),
        ("switch_pct", "switch", "dash", "#636EFA"),
        ("fsw_pct", "forced_switch", "dash", "#19D3F3"),
    ]
    fig = go.Figure()
    for run in has_actions["run"].unique():
        rd = has_actions[has_actions["run"] == run]
        label = RUN_META.get(run, run)
        for col, name, dash, color in eval_action_traces:
            if col in rd.columns and rd[col].notna().any():
                fig.add_trace(go.Scatter(
                    x=rd["steps"], y=rd[col] * 100,
                    mode="lines+markers", name=f"{label} - {name}",
                    legendgroup=run, line=dict(dash=dash, color=color),
                    marker=dict(size=4),
                ))
    fig.update_layout(height=400, yaxis_title="Action %", xaxis_title="Training Steps",
                      legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig, width="stretch")

# training health from eval data
health_cols = ["entropy", "kl"]
has_health = eval_df.dropna(subset=health_cols, how="all")
if not has_health.empty:
    st.subheader("Training Health (Eval Checkpoints)")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Entropy", "KL Divergence"))
    for run in has_health["run"].unique():
        rd = has_health[has_health["run"] == run]
        label = RUN_META.get(run, run)
        fig.add_trace(go.Scatter(
            x=rd["steps"], y=rd["entropy"], mode="lines+markers", name=label,
            marker=dict(size=4), legendgroup=run, showlegend=True,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=rd["steps"], y=rd["kl"], mode="lines+markers", name=label,
            marker=dict(size=4), legendgroup=run, showlegend=False,
        ), row=1, col=2)
    fig.update_layout(height=350, legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(title_text="Entropy", row=1, col=1)
    fig.update_yaxes(title_text="KL", row=1, col=2)
    st.plotly_chart(fig, width="stretch")

# action sequences from eval data
# color-coded by first action in the sequence
_SEQ_NAMES = {"d": "dmg", "s": "status", "e": "setup", "o": "other", "w": "switch", "f": "fsw"}
_SEQ_COLORS = {
    "d": ("#EF553B", "#C9402A", "#F47B66", "#A83525", "#D96050", "#B83020"),
    "s": ("#FFA15A", "#E08840", "#FFB87A", "#C07030", "#F09548", "#D07838"),
    "e": ("#AB63FA", "#9040E0", "#C490FF", "#7830C0", "#B878F0", "#6828A8"),
    "o": ("#B6E880", "#98C860", "#D0F8A0", "#80B040", "#A8D870", "#70A030"),
    "w": ("#636EFA", "#4855D8", "#8090FF", "#3540B0", "#5868E8", "#2830A0"),
    "f": ("#19D3F3", "#10B0D0", "#50E0F8", "#0890A8", "#30C8E8", "#0678A0"),
}
_CATS = ("d", "s", "e", "o", "w", "f")
SEQ_CONFIG = {}
for i, a in enumerate(_CATS):
    for j, b in enumerate(_CATS):
        key = a + b
        label = f"{_SEQ_NAMES[a]}->{_SEQ_NAMES[b]}"
        color = _SEQ_COLORS[a][j]
        SEQ_CONFIG[key] = (label, color)
seq_cols = [c for c in SEQ_CONFIG if c in eval_df.columns]
has_seq = eval_df.dropna(subset=seq_cols, how="all") if seq_cols else pd.DataFrame()
if not has_seq.empty:
    st.subheader("Action Sequences (Latest Eval)")
    latest = has_seq.groupby("run").last().reset_index()
    fig = go.Figure()
    for col in seq_cols:
        label, color = SEQ_CONFIG[col]
        fig.add_trace(go.Bar(
            x=[RUN_META.get(r, r) for r in latest["run"]],
            y=latest[col] * 100, name=label,
            marker_color=color,
        ))
    fig.update_layout(barmode="group", height=400, yaxis_title="Sequence %",
                      legend=dict(orientation="h", y=-0.3))
    st.plotly_chart(fig, width="stretch")

# ============================================================
# PBT (Population-Based Training)
# ============================================================
if PBT_CSV.exists():
    pbt_df = pd.read_csv(PBT_CSV)
    # support both old (score) and new (composite) CSV formats
    if "composite" in pbt_df.columns:
        pbt_df["score"] = pbt_df["composite"].astype(float)
    if not pbt_df.empty and "score" in pbt_df.columns:
        st.header("Population-Based Training")

        # best composite per generation
        gen_best = pbt_df.groupby("gen").agg(
            best=("score", "max"),
            mean=("score", "mean"),
            worst=("score", "min"),
        ).reset_index()

        st.subheader("Composite Score by Generation")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=gen_best["gen"], y=gen_best["best"] * 100,
            mode="lines+markers", name="Best", line=dict(color="#00CC96"),
        ))
        fig.add_trace(go.Scatter(
            x=gen_best["gen"], y=gen_best["mean"] * 100,
            mode="lines+markers", name="Mean", line=dict(color="#636EFA"),
        ))
        fig.add_trace(go.Scatter(
            x=gen_best["gen"], y=gen_best["worst"] * 100,
            mode="lines+markers", name="Worst", line=dict(color="#EF553B", dash="dash"),
        ))
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(height=400, yaxis_title="Composite %",
                          xaxis_title="Generation")
        st.plotly_chart(fig, width="stretch")

        # per-baseline win rates for best member each gen
        if "vs_maxdmg" in pbt_df.columns:
            best_per_gen = pbt_df.loc[pbt_df.groupby("gen")["score"].idxmax()]
            st.subheader("Best Member Win Rates by Baseline")
            fig = go.Figure()
            for col, name, color in [
                ("vs_maxdmg", "MaxDmg", "#EF553B"),
                ("vs_smart", "Smart", "#636EFA"),
                ("vs_crystal", "Crystal", "#00CC96"),
            ]:
                if col in best_per_gen.columns:
                    fig.add_trace(go.Scatter(
                        x=best_per_gen["gen"],
                        y=best_per_gen[col].astype(float) * 100,
                        mode="lines+markers", name=name,
                        line=dict(color=color),
                    ))
            fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
            fig.update_layout(height=400, yaxis_title="Win %",
                              xaxis_title="Generation")
            st.plotly_chart(fig, width="stretch")

        # per-member trajectories
        st.subheader("Member Trajectories")
        fig = go.Figure()
        for mid in sorted(pbt_df["member"].unique()):
            md = pbt_df[pbt_df["member"] == mid]
            fig.add_trace(go.Scatter(
                x=md["gen"], y=md["score"] * 100,
                mode="lines+markers", name=f"M{mid}",
                marker=dict(size=4),
            ))
        fig.add_hline(y=50, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(height=400, yaxis_title="Composite %",
                          xaxis_title="Generation")
        st.plotly_chart(fig, width="stretch")

        # hyperparam evolution
        st.subheader("Hyperparameter Evolution (Best Member)")
        best_member = pbt_df.loc[pbt_df.groupby("gen")["score"].idxmax()]
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=("Learning Rate", "Entropy Coef",
                                            "Gamma", "Clip Range"))
        fig.add_trace(go.Scatter(
            x=best_member["gen"], y=best_member["lr"],
            mode="lines+markers", name="lr",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=best_member["gen"], y=best_member["ent_coef"],
            mode="lines+markers", name="ent",
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            x=best_member["gen"], y=best_member["gamma"],
            mode="lines+markers", name="gamma",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=best_member["gen"], y=best_member["clip_range"],
            mode="lines+markers", name="clip",
        ), row=2, col=2)
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")

        with st.expander("Raw PBT Data"):
            st.dataframe(pbt_df, width="stretch")

# ---- Raw data ----
with st.expander("Raw Eval Data"):
    st.dataframe(eval_df, width="stretch")

if live_df is not None:
    with st.expander("Raw Live Data"):
        st.dataframe(live_df, width="stretch")

# ---- Auto-refresh ----
if refresh_map[refresh]:
    time.sleep(refresh_map[refresh])
    st.rerun()
