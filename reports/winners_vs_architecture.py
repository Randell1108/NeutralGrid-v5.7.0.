"""Interactive comparison: fast winners (<7h) vs rest of dataset vs current model defaults.

Reads data/new_expired_bots.xlsx and emits reports/winners_vs_architecture.html
(self-contained Plotly HTML dashboard).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "new_expired_bots.xlsx"
OUT = ROOT / "reports" / "winners_vs_architecture.html"

df = pd.read_excel(SRC)
win = df[(df["total_profit_usdt"] > 0) & (df["duration_hours"] < 7)].copy()
rest = df[~((df["total_profit_usdt"] > 0) & (df["duration_hours"] < 7))].copy()

FEATS = [
    ("duration_hours", "Duration (h)"),
    ("pnl_pct", "PnL %"),
    ("profit_factor", "Profit Factor"),
    ("grids_count", "Grids"),
    ("grid_spacing_pct", "Grid spacing %"),
    ("range_size_pct", "Range size %"),
    ("adx_1h", "ADX 1h"),
    ("adx_15m", "ADX 15m"),
    ("adx_5m", "ADX 5m"),
    ("rsi_15m", "RSI 15m"),
    ("bb_width", "BB width"),
    ("ema_crosses_5m", "EMA crosses 5m"),
    ("vwap_crosses_5m", "VWAP crosses 5m"),
    ("total_trades", "Total trades"),
]

# ---- Dashboard layout: 4 panels ----
fig = make_subplots(
    rows=3,
    cols=2,
    specs=[
        [{"type": "bar", "colspan": 2}, None],
        [{"type": "box"}, {"type": "box"}],
        [{"type": "scatter", "colspan": 2}, None],
    ],
    subplot_titles=(
        "Median-feature delta — Winners &lt;7h vs rest (current architecture defaults annotated)",
        "Regime features (ADX 1h / range_size / bb_width)",
        "Grid design (grids_count / spacing / trades)",
        "Grids vs Spacing — bubble = PnL%, color = group",
    ),
    vertical_spacing=0.11,
    row_heights=[0.26, 0.34, 0.40],
)

# Panel 1: median delta bar
meds = pd.DataFrame(
    {
        "feature": [lbl for _, lbl in FEATS],
        "winners": [win[c].median() for c, _ in FEATS],
        "rest": [rest[c].median() for c, _ in FEATS],
    }
)
meds["delta_pct"] = (meds["winners"] - meds["rest"]) / meds["rest"].abs().replace(0, 1) * 100
meds = meds.sort_values("delta_pct")
colors = ["#d62728" if v < 0 else "#2ca02c" for v in meds["delta_pct"]]
fig.add_trace(
    go.Bar(
        y=meds["feature"],
        x=meds["delta_pct"],
        orientation="h",
        marker_color=colors,
        text=[
            f"win={w:.3g} | rest={r:.3g}"
            for w, r in zip(meds["winners"], meds["rest"])
        ],
        hovertemplate="<b>%{y}</b><br>Δ median: %{x:.1f}%<br>%{text}<extra></extra>",
        name="Δ median %",
        showlegend=False,
    ),
    row=1,
    col=1,
)

# Panel 2: regime features — side-by-side boxplots
for i, (col, lbl) in enumerate([("adx_1h", "ADX 1h"), ("range_size_pct", "Range %"), ("bb_width", "BB width")]):
    fig.add_trace(
        go.Box(y=win[col].dropna(), name=f"W · {lbl}", marker_color="#2ca02c", boxmean=True, legendgroup="w", showlegend=(i == 0), legendgrouptitle_text="Winners <7h"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Box(y=rest[col].dropna(), name=f"R · {lbl}", marker_color="#7f7f7f", boxmean=True, legendgroup="r", showlegend=(i == 0), legendgrouptitle_text="Rest"),
        row=2, col=1,
    )

# Panel 3: grid design boxplots
for i, (col, lbl) in enumerate([("grids_count", "Grids"), ("grid_spacing_pct", "Spacing %"), ("total_trades", "Trades")]):
    fig.add_trace(
        go.Box(y=win[col].dropna(), name=f"W · {lbl}", marker_color="#1f77b4", boxmean=True, showlegend=False),
        row=2, col=2,
    )
    fig.add_trace(
        go.Box(y=rest[col].dropna(), name=f"R · {lbl}", marker_color="#c7c7c7", boxmean=True, showlegend=False),
        row=2, col=2,
    )

# Panel 4: scatter grids_count vs spacing, sized by PnL%, colored by group
def _bubble(sub: pd.DataFrame, name: str, color: str) -> go.Scatter:
    size = sub["pnl_pct"].clip(lower=0.3).fillna(0.3) * 3
    return go.Scatter(
        x=sub["grids_count"],
        y=sub["grid_spacing_pct"],
        mode="markers",
        marker=dict(size=size, color=color, opacity=0.7, line=dict(width=0.5, color="#222")),
        name=name,
        customdata=sub[["symbol", "duration_hours", "pnl_pct", "adx_1h", "trend_structure"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "grids=%{x} · spacing=%{y:.3f}%<br>"
            "dur=%{customdata[1]:.2f}h · pnl=%{customdata[2]:.2f}%<br>"
            "adx_1h=%{customdata[3]:.1f} · trend=%{customdata[4]}<extra></extra>"
        ),
    )

fig.add_trace(_bubble(win, "Winners <7h", "#2ca02c"), row=3, col=1)
fig.add_trace(_bubble(rest, "Rest", "#7f7f7f"), row=3, col=1)

# Annotate architecture defaults on panel 4
fig.add_vline(x=10, line=dict(dash="dash", color="#ff7f0e"), annotation_text="typical scanner default ~10 grids", annotation_position="top", row=3, col=1)
fig.add_hline(y=1.0, line=dict(dash="dash", color="#ff7f0e"), annotation_text="default spacing ~1%", annotation_position="right", row=3, col=1)

fig.update_xaxes(title_text="Δ median % (winners − rest)", row=1, col=1)
fig.update_xaxes(title_text="grids_count", row=3, col=1)
fig.update_yaxes(title_text="grid_spacing_pct", row=3, col=1)

fig.update_layout(
    title=dict(
        text=(
            "<b>Fast winners (&lt;7h, profitable)</b> vs rest — 41 winners / 121 rest · 162 total<br>"
            "<span style='font-size:12px;color:#666'>"
            "Architecture context: Stage B gates require range_prob ≥ threshold (HMM), "
            "scanner defaults favor low-ADX range regimes with ~10 grids at ~1% spacing."
            "</span>"
        ),
        x=0.02,
    ),
    height=1200,
    boxmode="group",
    template="plotly_white",
    margin=dict(l=80, r=40, t=120, b=60),
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.write_html(OUT, include_plotlyjs="cdn", full_html=True)
print(f"Wrote {OUT}")
