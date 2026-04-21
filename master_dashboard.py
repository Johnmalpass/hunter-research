"""HUNTER master dashboard.

Light, editorial, interactive. Six tabs: Overview, Corpus, Graph, Hypotheses,
Study, Operations. Read-only.

Run:
    streamlit run master_dashboard.py
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from itertools import combinations
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent
DB = ROOT / "hunter.db"
MANIFEST = ROOT / "preregistration.json"
GOALS = ROOT / "goals.json"
PRED_HTML = ROOT / "docs" / "index.html"

st.set_page_config(
    page_title="HUNTER",
    layout="wide",
    page_icon=None,
    initial_sidebar_state="collapsed",
)

# ============================================================================
# Palette
# ============================================================================
BG        = "#fafaf9"   # warm white
CARD      = "#ffffff"
BORDER    = "#e7e5e4"
BORDER2   = "#d6d3d1"
INK       = "#1c1917"
INK2      = "#44403c"
MUTED     = "#78716c"
MUTED2    = "#a8a29e"
ACCENT    = "#b45309"   # deep warm amber
ACCENT2   = "#92400e"
SUCCESS   = "#166534"
WARNING   = "#a16207"
ERROR     = "#991b1b"
CHART_SEQ = [ACCENT, "#1e3a8a", "#166534", "#7c2d12", "#6b21a8", "#065f46", "#9f1239"]

PLOTLY_TMPL = go.layout.Template()
PLOTLY_TMPL.layout = go.Layout(
    font=dict(family="-apple-system, 'SF Pro Text', Inter, 'Segoe UI', sans-serif",
              color=INK, size=13),
    paper_bgcolor=CARD,
    plot_bgcolor=CARD,
    margin=dict(l=20, r=20, t=30, b=30),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER, zerolinecolor=BORDER),
    colorway=CHART_SEQ,
    hoverlabel=dict(bgcolor=CARD, bordercolor=BORDER2, font_color=INK, font_size=12),
)

# ============================================================================
# Typography + layout CSS
# ============================================================================
st.markdown(f"""
<style>
    /* App shell */
    .stApp {{ background: {BG}; }}
    html, body, [class*="st-"] {{
        color: {INK};
        font-family: -apple-system, "SF Pro Text", Inter, "Segoe UI", Helvetica, sans-serif;
    }}

    /* Remove the default cramped top padding, add breathing room */
    [data-testid="stAppViewContainer"] > .main > .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 6rem;
        padding-left: 3.5rem;
        padding-right: 3.5rem;
        max-width: 1400px;
    }}

    /* Headings */
    h1 {{
        font-size: 2rem; font-weight: 600; letter-spacing: -0.02em;
        margin: 0 0 6px 0; color: {INK};
    }}
    h2 {{
        font-size: 0.82rem; font-weight: 600; letter-spacing: 0.12em;
        text-transform: uppercase; color: {MUTED};
        margin: 48px 0 18px 0; padding-bottom: 10px;
        border-bottom: 1px solid {BORDER};
    }}
    h3 {{
        font-size: 1rem; font-weight: 600; color: {INK};
        margin: 24px 0 10px 0;
    }}
    p, li, div, span {{ font-size: 0.95rem; line-height: 1.55; color: {INK2}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {MUTED}; font-size: 0.85rem; line-height: 1.5;
    }}

    /* Header strip */
    .hdr {{
        display: flex; align-items: baseline; gap: 24px;
        margin-bottom: 8px;
    }}
    .hdr .title {{
        font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;
        color: {INK}; margin: 0;
    }}
    .hdr .sub {{
        font-size: 0.95rem; color: {MUTED}; font-weight: 400;
    }}
    .hdr-meta {{
        display: flex; justify-content: flex-end; align-items: center; gap: 16px;
        padding: 6px 0 26px 0;
    }}
    .hdr-meta .phase {{
        background: {CARD}; border: 1px solid {BORDER};
        padding: 6px 14px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 500;
        color: {ACCENT2}; letter-spacing: 0.06em; text-transform: uppercase;
    }}
    .hdr-meta .time {{
        font-size: 0.8rem; color: {MUTED};
        font-variant-numeric: tabular-nums;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0; border-bottom: 1px solid {BORDER};
        margin-bottom: 32px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: transparent; border-radius: 0;
        padding: 12px 28px; color: {MUTED};
        font-weight: 500; font-size: 0.9rem;
        border-bottom: 2px solid transparent;
        transition: color 0.15s, border-color 0.15s;
    }}
    .stTabs [data-baseweb="tab"]:hover {{ color: {INK}; }}
    .stTabs [aria-selected="true"] {{
        background: transparent; color: {ACCENT2};
        border-bottom: 2px solid {ACCENT};
        font-weight: 600;
    }}

    /* Hero metric card */
    .hero-row {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 10px 0 6px 0;
    }}
    .hero-card {{
        background: {CARD}; border: 1px solid {BORDER};
        border-radius: 10px; padding: 18px 20px;
        transition: border-color 0.15s, transform 0.1s;
    }}
    .hero-card:hover {{ border-color: {BORDER2}; }}
    .hero-card .label {{
        font-size: 0.72rem; color: {MUTED};
        text-transform: uppercase; letter-spacing: 0.08em;
        font-weight: 600; margin-bottom: 8px;
    }}
    .hero-card .value {{
        font-size: 1.75rem; font-weight: 600; color: {INK};
        font-variant-numeric: tabular-nums;
        line-height: 1.1;
    }}

    /* Metric (built-in st.metric) */
    [data-testid="stMetricValue"] {{
        color: {INK}; font-size: 1.4rem; font-weight: 600;
        font-variant-numeric: tabular-nums;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED}; font-size: 0.74rem;
        text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;
    }}
    [data-testid="stMetricDelta"] {{
        color: {MUTED}; font-size: 0.8rem; font-variant-numeric: tabular-nums;
    }}

    /* Tables */
    .stDataFrame, [data-testid="stDataFrame"] {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
        overflow: hidden;
    }}
    .stDataFrame table {{ font-variant-numeric: tabular-nums; font-size: 0.88rem; }}

    /* Expander */
    .stExpander {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 8px;
        margin: 10px 0;
    }}
    .stExpander summary {{
        color: {INK}; font-weight: 500; padding: 12px 16px;
    }}
    .stExpander summary:hover {{ color: {ACCENT2}; }}

    /* Inputs — search box, selectbox, slider */
    [data-testid="stTextInput"] input,
    [data-testid="stSelectbox"] > div > div,
    [data-baseweb="select"] > div {{
        background: {CARD} !important;
        border-color: {BORDER} !important;
        color: {INK} !important;
    }}
    [data-testid="stTextInput"] input:focus {{
        border-color: {ACCENT} !important;
        outline: none !important;
    }}

    /* HR */
    hr {{ border: none; border-top: 1px solid {BORDER}; margin: 40px 0; }}

    /* Divider row */
    .row-item {{
        background: {CARD}; border: 1px solid {BORDER};
        padding: 14px 18px; margin: 8px 0; border-radius: 8px;
        transition: border-color 0.15s, box-shadow 0.15s;
    }}
    .row-item:hover {{
        border-color: {BORDER2};
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }}
    .row-item .score-pill {{
        display: inline-block;
        background: {ACCENT}; color: white;
        padding: 2px 10px; border-radius: 4px;
        font-variant-numeric: tabular-nums; font-weight: 600;
        font-size: 0.85rem; margin-right: 12px;
    }}
    .row-item .title {{ color: {INK}; font-weight: 500; font-size: 0.95rem; }}
    .row-item .meta {{ color: {MUTED}; font-size: 0.8rem; margin-top: 6px; }}
    .row-item .body {{ color: {INK2}; margin-top: 10px; font-size: 0.9rem; line-height: 1.6; }}

    /* Key-value strip (used for pre-reg integrity) */
    .kv-row {{
        display: flex; justify-content: space-between;
        padding: 10px 0; border-bottom: 1px solid {BORDER};
        font-size: 0.88rem;
    }}
    .kv-row:last-child {{ border-bottom: none; }}
    .kv-row .k {{ color: {MUTED}; }}
    .kv-row .v {{ color: {INK}; font-variant-numeric: tabular-nums; font-family: ui-monospace, "SF Mono", monospace; }}

    /* Status banner */
    .banner {{
        padding: 12px 18px; border-radius: 8px;
        margin: 16px 0; font-size: 0.9rem;
        border: 1px solid transparent;
    }}
    .banner.ok   {{ background: #f0fdf4; border-color: #bbf7d0; color: {SUCCESS}; }}
    .banner.warn {{ background: #fefce8; border-color: #fde68a; color: {WARNING}; }}
    .banner.err  {{ background: #fef2f2; border-color: #fecaca; color: {ERROR}; }}
    .banner.neutral {{ background: {CARD}; border-color: {BORDER}; color: {INK2}; }}

    /* Notifications (st.info/success/error) — neutralise to our palette */
    [data-testid="stNotification"] {{
        border-radius: 8px; font-size: 0.9rem;
    }}

    /* Code inline */
    code {{
        background: #f5f5f4; color: {ACCENT2};
        padding: 2px 6px; border-radius: 4px;
        font-size: 0.86rem; font-family: ui-monospace, "SF Mono", monospace;
    }}

    /* Cycle activity row (monospace line-style) */
    .cycle-line {{
        font-family: ui-monospace, "SF Mono", monospace;
        font-size: 0.82rem; padding: 4px 0;
        border-bottom: 1px dashed {BORDER};
    }}
    .cycle-line:last-child {{ border-bottom: none; }}
    .cycle-line .ts {{ color: {MUTED}; margin-right: 10px; }}
    .cycle-line .ok {{ color: {SUCCESS}; font-weight: 600; }}
    .cycle-line .warn {{ color: {WARNING}; font-weight: 600; }}
    .cycle-line .err {{ color: {ERROR}; font-weight: 600; }}
    .cycle-line .domain {{ color: {INK}; margin-left: 10px; }}

    /* Footer */
    footer, [data-testid="stFooter"] {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DB helpers
# ============================================================================
@st.cache_data(ttl=30, show_spinner=False)
def sql(q: str, params: tuple = ()) -> pd.DataFrame:
    try:
        conn = sqlite3.connect(DB)
        df = pd.read_sql_query(q, conn, params=params)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=30, show_spinner=False)
def sql_one(q: str, params: tuple = ()):
    try:
        conn = sqlite3.connect(DB)
        row = conn.execute(q, params).fetchone()
        conn.close()
        return row[0] if row else None
    except Exception:
        return None


def fmt(n) -> str:
    if n is None:
        return "—"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return str(n)


def empty_state(msg: str) -> None:
    st.markdown(
        f"<div class='banner neutral'>{msg}</div>",
        unsafe_allow_html=True,
    )


def hero_tile(label: str, value) -> str:
    return (
        f"<div class='hero-card'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value'>{fmt(value)}</div>"
        f"</div>"
    )


def apply_plotly_theme(fig):
    fig.update_layout(template=PLOTLY_TMPL)
    return fig


# ============================================================================
# Header
# ============================================================================
now = datetime.now()

try:
    from timeline import current_phase, next_phase  # noqa
    _p = current_phase()
    _nxt = next_phase()
    if _p is not None:
        phase_label = f"Phase {_p.id} · {_p.name}"
        phase_sub = f"{_p.days_remaining()} days remaining"
    elif _nxt is not None:
        phase_label = f"Pre-phase · {_nxt.name}"
        phase_sub = f"starts in {(_nxt.start - now.date()).days} days"
    else:
        phase_label = "No active phase"
        phase_sub = ""
except Exception:
    phase_label = "Operational"
    phase_sub = ""

col_h, col_m = st.columns([3, 2])
with col_h:
    st.markdown(
        f"""
<div class='hdr'>
    <span class='title'>HUNTER</span>
</div>
<div class='sub' style='color:{MUTED}; font-size:0.95rem; margin-top:-4px;'>
    Cross-silo research instrument · John Malpass · University College Dublin · 2026
</div>
        """,
        unsafe_allow_html=True,
    )
with col_m:
    st.markdown(
        f"""
<div class='hdr-meta'>
    <span class='phase'>{phase_label}</span>
    <span class='time'>{phase_sub}</span>
    <span class='time'>{now.strftime('%Y-%m-%d %H:%M:%S')}</span>
</div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================================
# Hero tiles (8 counts in 2 rows of 4)
# ============================================================================
facts_n = sql_one("SELECT COUNT(*) FROM raw_facts") or 0
anomalies_n = sql_one("SELECT COUNT(*) FROM anomalies") or 0
collisions_n = sql_one("SELECT COUNT(*) FROM collisions") or 0
chains_n = sql_one("SELECT COUNT(*) FROM chains") or 0
edges_n = sql_one("SELECT COUNT(*) FROM causal_edges") or 0
hypotheses_n = sql_one("SELECT COUNT(*) FROM hypotheses") or 0
survived_n = sql_one("SELECT COUNT(*) FROM hypotheses WHERE survived_kill=1") or 0
findings_n = sql_one("SELECT COUNT(*) FROM findings") or 0
cycles_n = sql_one("SELECT COUNT(*) FROM detected_cycles") or 0

row1_html = "<div class='hero-row'>" + "".join([
    hero_tile("Facts", facts_n),
    hero_tile("Anomalies", anomalies_n),
    hero_tile("Collisions", collisions_n),
    hero_tile("Chains", chains_n),
]) + "</div>"
row2_html = "<div class='hero-row'>" + "".join([
    hero_tile("Causal edges", edges_n),
    hero_tile("Hypotheses", hypotheses_n),
    hero_tile("Findings (≥65)", findings_n),
    hero_tile("Cycles", cycles_n),
]) + "</div>"
st.markdown(row1_html + row2_html, unsafe_allow_html=True)


# ============================================================================
# Tabs
# ============================================================================
tabs = st.tabs(["Overview", "Corpus", "Graph", "Hypotheses", "Study", "Operations"])


# ==========================================================================
# 1. OVERVIEW
# ==========================================================================
with tabs[0]:
    c_left, c_right = st.columns([3, 2], gap="large")

    with c_left:
        st.markdown("## Pipeline funnel")
        funnel_rows = pd.DataFrame([
            {"Stage": "1  Facts ingested",       "Count": facts_n,       "Yield": "—"},
            {"Stage": "2  Anomalies detected",   "Count": anomalies_n,
             "Yield": f"{(anomalies_n/max(1,facts_n))*100:.1f}% of facts"},
            {"Stage": "3  Collisions formed",    "Count": collisions_n,
             "Yield": f"{(collisions_n/max(1,anomalies_n))*100:.1f}% of anomalies"},
            {"Stage": "4  Hypotheses formed",    "Count": hypotheses_n,
             "Yield": f"{(hypotheses_n/max(1,collisions_n))*100:.1f}% of collisions"},
            {"Stage": "5  Survived kill phase",  "Count": survived_n,
             "Yield": f"{(survived_n/max(1,hypotheses_n))*100:.1f}% of hypotheses"},
            {"Stage": "6  Findings (score ≥ 65)", "Count": findings_n,
             "Yield": f"{(findings_n/max(1,survived_n))*100:.1f}% of survivors"},
        ])
        # Funnel bar chart (interactive)
        fig_funnel = apply_plotly_theme(go.Figure(go.Bar(
            x=funnel_rows["Count"], y=funnel_rows["Stage"],
            orientation="h", marker=dict(color=ACCENT, line=dict(color=ACCENT2, width=0)),
            text=funnel_rows["Count"].apply(fmt), textposition="outside",
            hovertemplate="%{y}<br><b>%{x:,}</b> rows<extra></extra>",
        )))
        fig_funnel.update_layout(height=320, margin=dict(l=0, r=40, t=20, b=10),
                                  yaxis=dict(autorange="reversed"), showlegend=False)
        st.plotly_chart(fig_funnel, use_container_width=True, config={"displayModeBar": False})

        with st.expander("Funnel table"):
            st.dataframe(funnel_rows, hide_index=True, use_container_width=True)

        st.markdown("## Recent cycle activity")
        recent = sql("""
            SELECT created_at, domain, status, error_message
            FROM cycle_logs ORDER BY id DESC LIMIT 10
        """)
        if not recent.empty:
            recent["created_at"] = pd.to_datetime(recent["created_at"])
            lines = []
            for _, r in recent.iterrows():
                cls = {"completed": "ok", "rate_limit": "warn"}.get(r["status"], "err")
                t = r["created_at"].strftime("%m-%d %H:%M")
                err = (str(r.get("error_message") or "")[:90]).replace("<", "&lt;")
                lines.append(
                    f"<div class='cycle-line'>"
                    f"<span class='ts'>{t}</span>"
                    f"<span class='{cls}'>{r['status']}</span>"
                    f"<span class='domain'>[{r['domain']}]</span>"
                    + (f"<span style='color:{MUTED}'> — {err}</span>" if err else "")
                    + "</div>"
                )
            st.markdown("".join(lines), unsafe_allow_html=True)
        else:
            empty_state("No cycles logged. Start with `python run.py live`.")

    with c_right:
        st.markdown("## System state")
        total_cycles = sql_one("SELECT COUNT(*) FROM cycle_logs") or 0
        last24 = sql_one("SELECT COUNT(*) FROM cycle_logs WHERE created_at >= datetime('now', '-1 day')") or 0
        last1h = sql_one("SELECT COUNT(*) FROM cycle_logs WHERE created_at >= datetime('now', '-1 hour')") or 0
        errors24 = sql_one("SELECT COUNT(*) FROM cycle_logs WHERE status != 'completed' AND created_at >= datetime('now', '-1 day')") or 0
        success_rate = (last24 - errors24) / max(1, last24)

        r1c1, r1c2 = st.columns(2)
        r1c1.metric("Cycles total", fmt(total_cycles))
        r1c2.metric("Cycles 24h", fmt(last24))
        r2c1, r2c2 = st.columns(2)
        r2c1.metric("Cycles 1h", fmt(last1h))
        r2c2.metric("24h success", f"{success_rate:.0%}",
                    delta=f"{errors24} errors" if errors24 else "clean",
                    delta_color="inverse" if errors24 else "normal")

        st.markdown("## Pre-registration integrity")
        if MANIFEST.exists():
            try:
                m = json.loads(MANIFEST.read_text())
                rows = [
                    ("Study", m.get("study_name", "—")),
                    ("Cutoff", m.get("corpus_cutoff", "—")),
                    ("Holdout start", m.get("holdout_start", "—")),
                    ("Frozen facts", fmt(m.get("corpus_fact_count"))),
                    ("Code hash", m.get("code_hash", "—")),
                    ("Fact-ID hash", (m.get("corpus_fact_id_hash") or "—")[:16] + "…"),
                    ("Locked at", (m.get("created_at") or "—")[:16]),
                ]
                st.markdown(
                    "".join(
                        f"<div class='kv-row'><span class='k'>{k}</span><span class='v'>{v}</span></div>"
                        for k, v in rows
                    ),
                    unsafe_allow_html=True,
                )

                try:
                    from preregister import verify_manifest  # noqa
                    v = verify_manifest()
                    if isinstance(v, dict) and v.get("status") == "ok":
                        st.markdown(
                            "<div class='banner ok'>Manifest verified — no drift detected.</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        drift = (v or {}).get("drift", "unknown") if isinstance(v, dict) else "unknown"
                        st.markdown(
                            f"<div class='banner err'>Drift detected: {drift}</div>",
                            unsafe_allow_html=True,
                        )
                except Exception:
                    pass
            except Exception as e:
                empty_state(f"Manifest unreadable: {e}")
        else:
            empty_state("No manifest. Run `python run.py preregister lock`.")


# ==========================================================================
# 2. CORPUS
# ==========================================================================
with tabs[1]:
    st.markdown("## Source-type distribution")
    sil = sql("""
        SELECT source_type AS silo, COUNT(*) AS n
        FROM raw_facts GROUP BY source_type ORDER BY n DESC
    """)
    if not sil.empty:
        fig = apply_plotly_theme(px.bar(sil, x="silo", y="n",
                                          color_discrete_sequence=[ACCENT]))
        fig.update_layout(height=340, xaxis_title=None, yaxis_title="facts",
                          showlegend=False, xaxis_tickangle=-35)
        fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,} facts<extra></extra>")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        empty_state("No facts ingested yet.")

    st.markdown("## Fact browser")
    c_search, c_filter = st.columns([3, 1])
    with c_search:
        search_q = st.text_input("Search facts (title + content)", "",
                                 placeholder="e.g. CMBS, OSHA, pharma approval",
                                 label_visibility="collapsed")
    with c_filter:
        silo_options = ["all"] + (sil["silo"].tolist() if not sil.empty else [])
        silo_filter = st.selectbox("Silo", silo_options, label_visibility="collapsed")

    where_parts, where_args = [], []
    if silo_filter and silo_filter != "all":
        where_parts.append("source_type = ?")
        where_args.append(silo_filter)
    if search_q.strip():
        where_parts.append("(title LIKE ? OR raw_content LIKE ?)")
        where_args.extend([f"%{search_q}%", f"%{search_q}%"])
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    browse = sql(f"""
        SELECT id, source_type AS silo, country, date_of_fact AS date,
               substr(title, 1, 140) AS title
        FROM raw_facts {where_clause}
        ORDER BY COALESCE(date_of_fact, ingested_at) DESC
        LIMIT 200
    """, tuple(where_args))
    if not browse.empty:
        st.caption(f"{len(browse)} facts shown (limited to 200)")
        st.dataframe(browse, hide_index=True, use_container_width=True, height=420)
    else:
        empty_state("No facts match those filters.")

    st.markdown("## Model-field extractions")
    mf = sql("""
        SELECT field_type, COUNT(*) AS n
        FROM fact_model_fields GROUP BY field_type ORDER BY n DESC
    """)
    if not mf.empty:
        c1, c2 = st.columns([1, 2], gap="large")
        with c1:
            fig = apply_plotly_theme(px.pie(mf, names="field_type", values="n",
                                              color_discrete_sequence=CHART_SEQ,
                                              hole=0.55))
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=10),
                              showlegend=True, legend=dict(orientation="v", x=1.02, y=0.5))
            fig.update_traces(hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with c2:
            top_meth = sql("""
                SELECT field_value AS methodology, COUNT(*) AS n
                FROM fact_model_fields
                WHERE field_type = 'methodology'
                GROUP BY field_value ORDER BY n DESC LIMIT 15
            """)
            st.markdown("### Top 15 methodologies named across facts")
            if not top_meth.empty:
                top_meth["methodology"] = top_meth["methodology"].str[:90]
                st.dataframe(top_meth, hide_index=True, use_container_width=True)
    else:
        empty_state("No model-field extractions yet.")


# ==========================================================================
# 3. GRAPH
# ==========================================================================
with tabs[2]:
    g_edges, g_chains, g_cycles, g_coll = st.tabs(
        ["Causal edges", "Chains", "Cycles", "Collision pairs"]
    )

    with g_edges:
        st.markdown("## Directed edges with named transmission pathway")
        edges = sql("""
            SELECT id, cause_node, effect_node, relationship_type,
                   confidence, source_type, domain
            FROM causal_edges
        """)
        if not edges.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("Edges total", fmt(len(edges)))
            m2.metric("Distinct nodes",
                      fmt(edges[["cause_node", "effect_node"]].stack().nunique()))
            m3.metric("Mean confidence", f"{edges['confidence'].mean():.2f}")

            deg_out = edges.groupby("cause_node").size().reset_index(name="out").rename(columns={"cause_node": "node"})
            deg_in = edges.groupby("effect_node").size().reset_index(name="in_").rename(columns={"effect_node": "node"})
            deg = deg_out.merge(deg_in, on="node", how="outer").fillna(0)
            deg["total"] = deg["out"] + deg["in_"]
            deg = deg.sort_values("total", ascending=False).head(15)
            deg["node_short"] = deg["node"].str[:80]

            st.markdown("### Top 15 hub nodes (by total degree)")
            fig = apply_plotly_theme(go.Figure(go.Bar(
                x=deg["total"], y=deg["node_short"], orientation="h",
                marker=dict(color=ACCENT),
                hovertemplate="<b>%{y}</b><br>in: %{customdata[0]}<br>out: %{customdata[1]}<br>total: %{x}<extra></extra>",
                customdata=deg[["in_", "out"]].values,
            )))
            fig.update_layout(height=480, margin=dict(l=0, r=40, t=10, b=20),
                              yaxis=dict(autorange="reversed"), showlegend=False)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            with st.expander(f"All edges ({len(edges)})"):
                edges_show = edges.copy()
                edges_show["cause_node"] = edges_show["cause_node"].str[:90]
                edges_show["effect_node"] = edges_show["effect_node"].str[:90]
                st.dataframe(edges_show, hide_index=True, use_container_width=True, height=460)
        else:
            empty_state("No edges yet. Edges form during ingest + collision cycles.")

    with g_chains:
        st.markdown("## Multi-link causal chains")
        chains = sql("""
            SELECT id, collision_id, chain_length, num_domains, domains_traversed
            FROM chains ORDER BY chain_length DESC, created_at DESC LIMIT 200
        """)
        if not chains.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric("Chains total", fmt(len(chains)))
            m2.metric("Mean length", f"{chains['chain_length'].mean():.1f}")
            m3.metric("Max length", fmt(chains["chain_length"].max()))

            len_dist = chains["chain_length"].value_counts().sort_index().reset_index()
            len_dist.columns = ["length", "n"]
            fig = apply_plotly_theme(px.bar(len_dist, x="length", y="n",
                                              color_discrete_sequence=[ACCENT]))
            fig.update_layout(height=240, showlegend=False)
            fig.update_traces(hovertemplate="length %{x}<br><b>%{y}</b> chains<extra></extra>")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.dataframe(chains, hide_index=True, use_container_width=True, height=360)
        else:
            empty_state("No chains yet.")

    with g_cycles:
        st.markdown("## Detected epistemic cycles")
        st.caption("Closed loops A → B → … → A identified by Tarjan SCC over the causal graph.")
        cycles = sql("""
            SELECT id, cycle_type, nodes, domains,
                   reinforcement_strength, persistence_estimate,
                   detected_date, age_days
            FROM detected_cycles
            ORDER BY reinforcement_strength DESC
        """)
        if not cycles.empty:
            tcount = cycles["cycle_type"].value_counts().reset_index()
            tcount.columns = ["type", "n"]
            c1, c2 = st.columns([1, 2], gap="large")
            with c1:
                fig = apply_plotly_theme(px.bar(tcount, x="n", y="type", orientation="h",
                                                  color_discrete_sequence=[ACCENT]))
                fig.update_layout(height=260, showlegend=False,
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with c2:
                st.markdown("### Cycle details")
                for _, r in cycles.iterrows():
                    try:
                        nodes = json.loads(r["nodes"] or "[]")
                    except Exception:
                        nodes = []
                    with st.expander(
                        f"#{r['id']}  ·  {r['cycle_type']}  ·  "
                        f"{len(nodes)} nodes  ·  strength {r['reinforcement_strength']:.2f}"
                    ):
                        for i, n in enumerate(nodes):
                            if isinstance(n, dict):
                                dom = n.get('domain', '?')
                                meth = n.get('methodology', '')
                                st.markdown(f"**{i+1}.** {dom[:120]}")
                                if meth:
                                    st.caption(meth[:220])
                            else:
                                st.markdown(f"**{i+1}.** {str(n)[:120]}")
                        st.caption(
                            f"detected {str(r['detected_date'])[:10]} · "
                            f"est persistence {r.get('persistence_estimate') or 0:.0f}d"
                        )
        else:
            empty_state("No cycles detected. Run `python cycle_detector.py run`.")

    with g_coll:
        st.markdown("## Cross-silo collision pairs")
        fire_df = sql("""
            SELECT source_types FROM collisions
            WHERE source_types IS NOT NULL AND source_types != ''
        """)
        if not fire_df.empty:
            pair_counts: dict[str, int] = {}
            for _, r in fire_df.iterrows():
                s = r["source_types"]
                try:
                    types = json.loads(s) if isinstance(s, str) and s.startswith("[") else s.split(",")
                except Exception:
                    types = []
                types = sorted({t.strip() for t in types if t and str(t).strip()})
                if len(types) < 2:
                    continue
                for a, b in combinations(types, 2):
                    key = f"{a} × {b}"
                    pair_counts[key] = pair_counts.get(key, 0) + 1
            if pair_counts:
                top = pd.DataFrame([
                    {"pair": k, "n": v}
                    for k, v in sorted(pair_counts.items(), key=lambda x: -x[1])[:25]
                ])
                fig = apply_plotly_theme(go.Figure(go.Bar(
                    x=top["n"], y=top["pair"], orientation="h",
                    marker=dict(color=ACCENT),
                    hovertemplate="<b>%{y}</b><br>%{x} collisions<extra></extra>",
                )))
                fig.update_layout(height=620, margin=dict(l=0, r=40, t=10, b=10),
                                  yaxis=dict(autorange="reversed"), showlegend=False)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            empty_state("No collisions yet.")


# ==========================================================================
# 4. HYPOTHESES
# ==========================================================================
with tabs[3]:
    h_top, h_all, h_portfolio, h_board = st.tabs(
        ["Top findings", "All hypotheses", "Paper-trade positions", "Public board"]
    )

    with h_top:
        st.markdown("## Top-scoring findings (diamond ≥ 65)")
        diamonds = sql("""
            SELECT id, title, score, domain, confidence, summary, created_at
            FROM findings ORDER BY score DESC LIMIT 100
        """)
        if not diamonds.empty:
            c_thr, c_dom = st.columns([1, 1])
            with c_thr:
                min_score = st.slider("Minimum score", 65, 100, 65, 1)
            with c_dom:
                dom_opts = ["all"] + sorted(diamonds["domain"].dropna().unique().tolist())
                dom_filter = st.selectbox("Domain", dom_opts, key="diamond_dom")

            filt = diamonds[diamonds["score"] >= min_score]
            if dom_filter != "all":
                filt = filt[filt["domain"] == dom_filter]
            st.caption(f"{len(filt)} findings match")

            if not filt.empty:
                c1, c2 = st.columns([1, 3], gap="large")
                with c1:
                    hist = filt["score"].value_counts().sort_index().reset_index()
                    hist.columns = ["score", "n"]
                    fig = apply_plotly_theme(px.bar(hist, x="score", y="n",
                                                      color_discrete_sequence=[ACCENT]))
                    fig.update_layout(height=260, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                with c2:
                    for _, r in filt.head(15).iterrows():
                        score = int(r["score"])
                        st.markdown(f"""
<div class='row-item'>
    <span class='score-pill'>{score}</span>
    <span class='title'>{(r['title'] or '')[:140]}</span>
    <div class='meta'>{r['domain']} · {r['confidence']} · {str(r['created_at'])[:10]}</div>
    <div class='body'>{(r['summary'] or '')[:320]}</div>
</div>
                        """, unsafe_allow_html=True)
        else:
            empty_state("No findings yet. Entries appear once HUNTER has run collision cycles "
                         "and produced hypotheses scoring ≥ 65.")

    with h_all:
        st.markdown("## All hypotheses with completed adversarial review")
        all_h = sql("""
            SELECT h.id, h.diamond_score AS score, h.confidence,
                   h.survived_kill, c.num_domains, c.source_types,
                   substr(h.hypothesis_text, 1, 220) AS thesis,
                   h.time_window_days AS window_d,
                   h.created_at
            FROM hypotheses h LEFT JOIN collisions c ON c.id = h.collision_id
            ORDER BY h.diamond_score DESC
        """)
        if not all_h.empty:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Hypotheses", fmt(len(all_h)))
            m2.metric("Survived", fmt(int((all_h["survived_kill"] == 1).sum())))
            m3.metric("Killed", fmt(int((all_h["survived_kill"] == 0).sum())))
            m4.metric("Mean score", f"{all_h['score'].mean():.1f}")

            c_surv, c_nd = st.columns([1, 1])
            with c_surv:
                surv_filter = st.selectbox("Outcome",
                                            ["all", "survived", "killed"],
                                            key="hyp_surv")
            with c_nd:
                nd_filter = st.selectbox("Num silos",
                                          ["all"] + sorted(all_h["num_domains"].dropna().unique().tolist()),
                                          key="hyp_nd")
            filt = all_h.copy()
            if surv_filter == "survived":
                filt = filt[filt["survived_kill"] == 1]
            elif surv_filter == "killed":
                filt = filt[filt["survived_kill"] == 0]
            if nd_filter != "all":
                filt = filt[filt["num_domains"] == nd_filter]
            st.caption(f"{len(filt)} hypotheses shown")
            st.dataframe(filt, hide_index=True, use_container_width=True, height=520)
        else:
            empty_state("No hypotheses yet.")

    with h_portfolio:
        st.markdown("## Paper-trade positions")
        st.caption("Internal view only. Paper-trade tables are intentionally "
                    "withheld from the Zenodo v1 release; they're shown here so you "
                    "can track the private book.")
        positions = sql("""
            SELECT id, ticker, direction, entry_price, current_price,
                   pnl_pct, pnl_gbp, diamond_score, confidence, status,
                   entry_date, close_date, hypothesis_text
            FROM portfolio_positions WHERE ticker != 'LOGGED'
        """)
        if positions.empty:
            empty_state("No paper-trade positions.")
        else:
            opos = positions[positions["status"] == "open"]
            cpos = positions[positions["status"] == "closed"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Open", fmt(len(opos)))
            c2.metric("Closed", fmt(len(cpos)))
            if not cpos.empty:
                c3.metric("Avg closed P&L", f"{cpos['pnl_pct'].mean():+.2f}%")
                wins = int((cpos["pnl_pct"] > 0).sum())
                c4.metric("Win rate", f"{wins/max(1,len(cpos)):.0%}")
            else:
                c3.metric("Avg closed P&L", "—")
                c4.metric("Win rate", "—")

            st.markdown("### Open positions")
            if not opos.empty:
                show = opos[["ticker", "direction", "entry_price", "current_price",
                             "pnl_pct", "pnl_gbp", "diamond_score", "entry_date"]]
                st.dataframe(show, hide_index=True, use_container_width=True)
            else:
                empty_state("No open positions.")

    with h_board:
        st.markdown("## Public prediction board")
        st.caption("Every hypothesis scoring ≥ 65 posts with asset, direction, and resolution date. "
                    "Board is deliberately empty until the pre-registered summer run begins June 1.")
        if PRED_HTML.exists():
            st.markdown(
                "<div class='banner ok'>Board deployed → "
                "<code>johnmalpass.github.io/hunter-research/</code></div>",
                unsafe_allow_html=True,
            )
        try:
            from prediction_board import gather_predictions, compute_track_record  # noqa
            preds = gather_predictions(min_score=65)
            tr = compute_track_record(preds)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Posted", fmt(tr["total_predictions"]))
            c2.metric("Pending", fmt(tr["pending"]))
            c3.metric("Resolved", fmt(tr["resolved"]))
            c4.metric("Hit rate",
                      f"{tr['hit_rate']:.0%}" if tr.get("hit_rate") is not None else "—")
            c5.metric("Brier",
                      f"{tr['brier_score']}" if tr.get("brier_score") is not None else "—")
            if preds:
                df = pd.DataFrame([{
                    "ID": p["id"], "Score": p["diamond_score"],
                    "Status": p["status"], "Posted": p["posted_date"],
                    "Target": p["target_date"], "Days left": p["days_remaining"],
                    "Confidence": p["confidence"],
                    "Thesis": p["thesis_short"],
                } for p in preds])
                st.dataframe(df, hide_index=True, use_container_width=True)
        except Exception as e:
            st.caption(f"Board module not available: {e}")


# ==========================================================================
# 5. STUDY
# ==========================================================================
with tabs[4]:
    s_hyps, s_tests, s_layers = st.tabs(
        ["Pre-registered hypothesis tests", "Framework empirical tests", "13-layer evidence"]
    )

    with s_hyps:
        st.markdown("## Pre-registered hypothesis tests")
        ft = sql("""
            SELECT hypothesis_id, hypothesis_name, supports_hypothesis,
                   observation_value, measured_at
            FROM frontier_test_results ORDER BY measured_at DESC LIMIT 100
        """)
        if not ft.empty:
            latest = ft.drop_duplicates(subset=["hypothesis_id"], keep="first")
            m1, m2, m3 = st.columns(3)
            m1.metric("Tests measured", fmt(len(latest)))
            m2.metric("Supported", fmt(int((latest["supports_hypothesis"] == 1).sum())))
            m3.metric("Refuted", fmt(int((latest["supports_hypothesis"] == 0).sum())))
            st.dataframe(latest, hide_index=True, use_container_width=True, height=440)
        else:
            empty_state("No frontier-test results. Run `python run.py frontier all`.")

    with s_tests:
        st.markdown("## 1. Collision formula predictiveness")
        fv = sql("SELECT * FROM formula_validation ORDER BY date DESC LIMIT 1")
        if not fv.empty:
            row = fv.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Pearson r", f"{row.get('pearson_r', 0):+.3f}")
            c2.metric("Spearman ρ", f"{row.get('spearman_rho', 0):+.3f}")
            c3.metric("p-value", f"{row.get('p_value', 0):.3f}")
        else:
            empty_state("Run `python formula_validator.py write`.")

        st.markdown("## 2. Measured reinforcement and correction per silo")
        measured = sql("""
            SELECT source_type AS silo, reinforcement_measured, correction_measured,
                   persistence_ratio_measured, n_facts
            FROM measured_domain_params ORDER BY persistence_ratio_measured DESC
        """)
        if not measured.empty:
            st.dataframe(measured, hide_index=True, use_container_width=True, height=340)
        else:
            empty_state("Run `python reinforcement_measurer.py write`.")

        st.markdown("## 3. Half-life per silo vs framework prediction")
        hl = sql("""
            SELECT source_type AS silo, half_life_days, n_correction_events, n_observations
            FROM halflife_estimates ORDER BY half_life_days
        """)
        if not hl.empty:
            st.dataframe(hl, hide_index=True, use_container_width=True, height=320)
        else:
            empty_state("Run `python halflife_estimator.py write`.")

        st.markdown("## 4. Narrative strength vs kill survival")
        ns = sql("""
            SELECT narrative_strength, h.survived_kill, h.diamond_score
            FROM narrative_scores ns JOIN hypotheses h ON h.id = ns.hypothesis_id
        """)
        if not ns.empty:
            high = ns[ns["narrative_strength"] >= 0.6]
            low = ns[ns["narrative_strength"] < 0.4]
            c1, c2, c3 = st.columns(3)
            c1.metric("High-narrative n", len(high),
                      delta=f"{high['survived_kill'].mean():.0%} survival" if not high.empty else "—")
            c2.metric("Low-narrative n", len(low),
                      delta=f"{low['survived_kill'].mean():.0%} survival" if not low.empty else "—")
            if not high.empty and not low.empty:
                c3.metric("Uplift",
                          f"{(high['survived_kill'].mean() - low['survived_kill'].mean()):+.1%}")
        else:
            empty_state("Run `python narrative_detector.py write`.")

    with s_layers:
        st.markdown("## 13-layer theory-evidence matrix")
        layer_names = {
            1: "Translation Loss",      2: "Attention Topology",
            3: "Question Gap",          4: "Phase Transition",
            5: "Rate-Distortion",       6: "Market Incompleteness",
            7: "Depth-Value",           8: "Epistemic Cycles",
            9: "Cycle Hierarchy",       10: "Fractal Incompleteness",
            11: "Negative Space",       12: "Autopoiesis",
            13: "Observer-Dependent",
        }
        ev = sql("""
            SELECT layer, evidence_type, COUNT(*) AS n
            FROM theory_evidence GROUP BY layer, evidence_type
        """)
        rows = []
        for i in range(1, 14):
            le = ev[ev["layer"] == i] if not ev.empty else pd.DataFrame()
            direct = int(le[le["evidence_type"] == "direct"]["n"].sum()) if not le.empty else 0
            support = int(le[le["evidence_type"] == "supporting"]["n"].sum()) if not le.empty else 0
            challenge = int(le[le["evidence_type"] == "challenging"]["n"].sum()) if not le.empty else 0
            if challenge > direct and challenge > 0:
                state = "challenged"
            elif direct > 0:
                state = "direct"
            elif support > 0:
                state = "supporting"
            else:
                state = "empty"
            rows.append({
                "Layer": f"L{i:02d}", "Name": layer_names.get(i, "?"),
                "State": state, "Direct": direct,
                "Supporting": support, "Challenging": challenge,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ==========================================================================
# 6. OPERATIONS
# ==========================================================================
with tabs[5]:
    o_agents, o_runs, o_goals, o_reports = st.tabs(
        ["Theory agents", "Cycle history", "Goals", "Reports"]
    )

    with o_agents:
        st.markdown("## Theory-layer agents")
        st.caption("Seven agents attached to the orchestrator. Each writes its most "
                    "recent output into the tables shown. `idle` does not mean broken — "
                    "it means the agent's next scheduled slot hasn't arrived yet.")

        agent_specs = [
            ("TheoryTelemetry",           "theory_evidence",       "created_at", "per cycle"),
            ("DecayTracker",              "decay_tracking",        "recorded_at", "daily"),
            ("CycleDetector",             "detected_cycles",       "detected_date", "weekly"),
            ("CollisionFormulaValidator", "formula_validation",    "date", "weekly"),
            ("ChainDepthProfiler",        "chains",                "created_at", "weekly"),
            ("BacktestReconciler",        "backtest_results",      "created_at", "weekly"),
            ("ResidualEstimator",         "residual_tam",          "measured_at", "monthly"),
        ]
        agent_rows = []
        for name, table, ts_col, cadence in agent_specs:
            last_ts = sql_one(f"SELECT {ts_col} FROM {table} ORDER BY {ts_col} DESC LIMIT 1")
            n_rows = sql_one(f"SELECT COUNT(*) FROM {table}") or 0
            if last_ts:
                try:
                    dt_last = pd.to_datetime(last_ts).to_pydatetime()
                    age_h = (datetime.now() - dt_last).total_seconds() / 3600
                    if age_h < 24:
                        state = "active"
                    elif age_h < 24 * 7:
                        state = "recent"
                    elif age_h < 24 * 30:
                        state = "stale"
                    else:
                        state = "idle"
                    last_str = dt_last.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    state = "unknown"
                    last_str = str(last_ts)[:16]
            else:
                state = "empty"
                last_str = "—"
            agent_rows.append({
                "Agent": name,
                "Cadence": cadence,
                "Output table": table,
                "Rows": fmt(n_rows),
                "Last run": last_str,
                "State": state,
            })
        st.dataframe(pd.DataFrame(agent_rows), hide_index=True, use_container_width=True)

        st.markdown("## Orchestrator scheduling")
        st.markdown(f"""
<div class='banner neutral'>
Theory agents run on fixed intervals from <code>orchestrator.py</code>.
<code>DecayTracker</code> runs daily at the 24 h rollover. <code>CycleDetector</code>,
<code>CollisionFormulaValidator</code>, <code>ChainDepthProfiler</code>,
<code>BacktestReconciler</code> run weekly on Sundays. <code>ResidualEstimator</code>
runs monthly on the 1st. <code>TheoryTelemetry</code> runs inline during each
collision cycle. Start the orchestrator with <code>python run.py live</code>.
</div>
        """, unsafe_allow_html=True)

    with o_runs:
        st.markdown("## Cycle history (last 100)")
        runs = sql("""
            SELECT id, datetime(created_at) AS t, domain, status,
                   tokens_used, duration_seconds, error_message
            FROM cycle_logs ORDER BY id DESC LIMIT 100
        """)
        if not runs.empty:
            runs["t"] = pd.to_datetime(runs["t"])
            runs["hour"] = runs["t"].dt.floor("h")
            hourly = runs.groupby(["hour", "status"]).size().unstack(fill_value=0).reset_index()
            # Plotly stacked area
            fig = go.Figure()
            for status_col in [c for c in hourly.columns if c != "hour"]:
                color = {"completed": SUCCESS, "rate_limit": WARNING}.get(status_col, ERROR)
                fig.add_trace(go.Scatter(
                    x=hourly["hour"], y=hourly[status_col],
                    stackgroup="one", name=status_col,
                    line=dict(width=0), fillcolor=color,
                    hovertemplate=f"<b>{status_col}</b><br>%{{x}}<br>%{{y}} cycles<extra></extra>",
                ))
            apply_plotly_theme(fig)
            fig.update_layout(height=260, margin=dict(l=0, r=10, t=20, b=10),
                              xaxis_title=None, yaxis_title="cycles")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

            st.markdown("### Cycles (most recent 50)")
            show = runs.head(50)[["t", "domain", "status", "tokens_used",
                                    "duration_seconds", "error_message"]]
            st.dataframe(show, hide_index=True, use_container_width=True, height=420)
        else:
            empty_state("No runs yet.")

    with o_goals:
        st.markdown("## Self-improvement goals")
        if GOALS.exists():
            g = json.loads(GOALS.read_text())
            idx = g.get("current_goal_index", 0)
            goals = g.get("goals", [])
            if idx < len(goals):
                current = goals[idx]
                st.markdown(
                    f"<div class='banner ok'><b>Active goal:</b> {current.get('goal')}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"Target: {current.get('target')} · Measure: {current.get('measure')}")
                subgoals = current.get("subgoals", [])
                if subgoals:
                    with st.expander(f"Subgoals ({len(subgoals)})"):
                        for s in subgoals:
                            st.markdown(f"- {s}")
            with st.expander(f"All goals ({len(goals)})"):
                for i, gl in enumerate(goals):
                    marker = "done" if i < idx else ("active" if i == idx else "queued")
                    color = {"done": MUTED, "active": ACCENT, "queued": MUTED2}[marker]
                    st.markdown(
                        f"<div style='display:flex; gap:12px; padding:4px 0; font-size:0.88rem;'>"
                        f"<span style='color:{color}; width:64px; font-size:0.76rem; "
                        f"text-transform:uppercase; letter-spacing:0.08em;'>{marker}</span>"
                        f"<span>#{i}  {gl.get('goal')}</span></div>",
                        unsafe_allow_html=True,
                    )
        else:
            empty_state("No goals.json found.")

    with o_reports:
        st.markdown("## Overseer reports (most recent 5)")
        overseer = sql("""
            SELECT created_at, substr(report_text, 1, 600) AS preview
            FROM overseer_reports ORDER BY id DESC LIMIT 5
        """)
        if not overseer.empty:
            for _, r in overseer.iterrows():
                with st.expander(str(r["created_at"])[:16]):
                    st.markdown(r["preview"])
        else:
            empty_state("No overseer reports. Run `python targeting.py`.")

        st.markdown("## Daily summaries")
        daily = sql("""
            SELECT summary_date, total_cycles, total_findings, diamonds_found,
                   most_promising_thread
            FROM daily_summaries ORDER BY summary_date DESC LIMIT 14
        """)
        if not daily.empty:
            st.dataframe(daily, hide_index=True, use_container_width=True, height=400)
        else:
            empty_state("No daily summaries yet.")


# ============================================================================
# Footer
# ============================================================================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    f"<div style='color:{MUTED}; font-size:0.8rem; text-align:center; "
    f"padding: 16px 0;'>"
    f"HUNTER · {now.strftime('%Y-%m-%d %H:%M:%S')} · John Malpass · "
    f"University College Dublin · "
    f"<a href='https://github.com/Johnmalpass/hunter-research' "
    f"style='color:{ACCENT}; text-decoration:none;'>github</a> · "
    f"<a href='https://doi.org/10.5281/zenodo.19667567' "
    f"style='color:{ACCENT}; text-decoration:none;'>corpus DOI</a>"
    f"</div>",
    unsafe_allow_html=True,
)
