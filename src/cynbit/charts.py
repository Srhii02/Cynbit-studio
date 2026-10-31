"""Plotly figures used by the Streamlit dashboard."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

INDIGO, TEAL, AMBER, ROSE, SLATE = "#4F46E5", "#0D9488", "#D97706", "#E11D48", "#64748B"
FONT = dict(family="Inter, Segoe UI, sans-serif", size=13)


def _layout(fig: go.Figure, height: int = 320, **kw) -> go.Figure:
    opts = dict(height=height, margin=dict(l=10, r=10, t=40, b=10), font=FONT,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    opts.update(kw)
    fig.update_layout(**opts)
    return fig


def score_color(v: float) -> str:
    return TEAL if v >= 75 else INDIGO if v >= 60 else AMBER if v >= 45 else ROSE


def gauge(value: float, title: str = "Match score") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number={"suffix": "/100", "font": {"size": 34}},
        title={"text": title, "font": {"size": 14}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 0}, "bar": {"color": score_color(value), "thickness": 0.3},
               "bgcolor": "rgba(148,163,184,0.15)", "borderwidth": 0,
               "steps": [{"range": [0, 45], "color": "rgba(225,29,72,0.08)"},
                         {"range": [45, 60], "color": "rgba(217,119,6,0.08)"},
                         {"range": [60, 75], "color": "rgba(79,70,229,0.08)"},
                         {"range": [75, 100], "color": "rgba(13,148,136,0.10)"}]}))
    return _layout(fig, 250, margin=dict(l=40, r=45, t=50, b=0))


def component_bars(result) -> go.Figure:
    labels = ["Semantic", "Skills", "Experience", "Education"]
    vals = [result.semantic, result.skills, result.experience, result.education]
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker_color=[score_color(v) for v in vals],
                           text=[f"{v:.0f}" for v in vals], textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[0, 110], showgrid=False, visible=False)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 230, title="Score components (weights 40/35/15/10)")


def radar(components: dict[str, float], title: str = "Readiness profile") -> go.Figure:
    keys = list(components)
    vals = [components[k] for k in keys]
    fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=keys + keys[:1], fill="toself",
                                    line_color=INDIGO, fillcolor="rgba(79,70,229,0.25)"))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], showticklabels=True, tickfont_size=10),
                                 bgcolor="rgba(0,0,0,0)"), showlegend=False)
    return _layout(fig, 380, title=title, margin=dict(l=90, r=90, t=60, b=40))


def ranking_bars(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color in (("Semantic", INDIGO), ("Skills", TEAL), ("Experience", AMBER), ("Education", SLATE)):
        fig.add_bar(name=col, x=df["Role"], y=df[col], marker_color=color)
    fig.add_scatter(name="Overall", x=df["Role"], y=df["Overall"], mode="markers",
                    hovertemplate="%{x}: %{y:.0f}<extra>Overall</extra>",
                    marker=dict(size=16, color="#111827", symbol="diamond", line=dict(color="white", width=2)))
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=-0.25))
    fig.update_yaxes(range=[0, 110], title="Score")
    return _layout(fig, 420, title="Job comparison")


def skill_heatmap(matrix: pd.DataFrame) -> go.Figure:
    m = matrix.head(25)
    z = m.values
    text = [["✓" if v == 1 else "~" if v == 0.5 else "✗" if v == 0 else "" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=list(m.columns), y=list(m.index), text=text, texttemplate="%{text}",
                               colorscale=[[0, "#FECDD3"], [0.5, "#FDE68A"], [1, "#99F6E4"]], zmin=0, zmax=1,
                               showscale=False, hoverongaps=False, xgap=2, ygap=2))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, max(300, 24 * len(m) + 80), title="Skill demand matrix (✓ have · ~ related · ✗ missing)")


def coverage_bars(coverage: dict[str, float]) -> go.Figure:
    items = sorted(coverage.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker_color=[score_color(v) for _, v in items],
                           text=[f"{v:.0f}%" for _, v in items], textposition="outside", cliponaxis=False))
    fig.update_xaxes(range=[0, 115], visible=False)
    return _layout(fig, max(260, 34 * len(items) + 60), title="Skill coverage by category (across target jobs)")


def roadmap(recs) -> go.Figure:
    fig = go.Figure()
    for r in reversed(recs):
        color = ROSE if r.priority_label == "High" else AMBER if r.priority_label == "Medium" else SLATE
        fig.add_bar(x=[r.week_end - r.week_start + 1], y=[r.skill], base=[r.week_start - 1], orientation="h",
                    marker_color=color, hovertext=f"{r.est_hours} h · {r.priority_label} priority", showlegend=False)
    fig.update_xaxes(title="Week", dtick=1)
    return _layout(fig, max(260, 30 * len(recs) + 80), title="Learning roadmap")


def trend(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Scatter(x=list(range(1, len(df) + 1)), y=df["overall"], mode="lines+markers",
                               text=df["job_title"], line_color=INDIGO))
    fig.update_xaxes(title="Analysis #", dtick=1)
    fig.update_yaxes(range=[0, 100], title="Overall score")
    return _layout(fig, 280, title="Your match-score history")
