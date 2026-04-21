#!/usr/bin/env python3
"""HUNTER Unified Dashboard -- Intelligence + Research Factory + Portfolio.

Run with: streamlit run hunter_dashboard.py
"""

import json
import os
import statistics
import subprocess
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from database import (
    get_all_positions,
    get_closed_positions,
    get_collisions_list,
    get_connection,
    get_held_collisions_count,
    get_held_collisions_list,
    get_hypotheses_list,
    get_open_positions,
    get_portfolio_snapshots,
    get_portfolio_stats,
    get_raw_facts_stream,
    get_recent_anomalies,
    get_source_type_diversity_score,
    get_v2_dashboard_stats,
    init_db,
    toggle_reviewed,
)
from config import SOURCE_ICONS, CORE_SOURCE_TYPES
from targeting import suggest_firms, generate_domains_for_firm, run_overseer, chat_with_hunter
from database import (
    get_active_targets, save_target, remove_target,
    get_firm_suggestions, save_firm_suggestions,
    get_latest_overseer_report,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="HUNTER",
    page_icon="\U0001f48e",
    layout="wide",
    initial_sidebar_state="collapsed",
)

init_db()

# ---------------------------------------------------------------------------
# Professional white theme CSS -- matching PDF doc style
# Navy #1B2A4A | Accent #2E5090 | Light BG #F5F6F8 | Text #2D2D2D
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ===== BASE ===== */
    .stApp { background-color: #FAFBFC; color: #374151; }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stSidebarCollapsedControl"] { display: none !important; }
    .stMainBlockContainer { padding-top: 0 !important; }
    section.stMain > div { padding-top: 0 !important; }

    /* ===== TOP NAV BAR ===== */
    .top-nav { background: #1B2A4A; padding: 0; margin: -1rem -4rem 0 -4rem;
        display: flex; align-items: center; justify-content: space-between;
        border-bottom: 3px solid #2E5090; position: sticky; top: 0; z-index: 999; }
    .top-nav-inner { width: 100%; padding: 16px 40px;
        display: flex; align-items: center; justify-content: space-between; }
    .top-nav-brand { display: flex; align-items: center; gap: 10px; }
    .top-nav-brand-text { color: #FFFFFF !important; font-family: -apple-system, sans-serif;
        font-weight: 700; font-size: 1.4em; letter-spacing: 0.5px; }
    .top-nav-brand-sub { color: #94A3B8 !important; font-size: 0.85em; font-weight: 400; letter-spacing: 0.5px; }
    .top-nav-stats { display: flex; gap: 40px; align-items: center; }
    .top-nav-stat { text-align: center; }
    .top-nav-stat-value { color: #FFFFFF !important; font-size: 1.6em; font-weight: 700;
        font-family: -apple-system, sans-serif; line-height: 1.2; }
    .top-nav-stat-label { color: #94A3B8 !important; font-size: 0.7em; text-transform: uppercase;
        letter-spacing: 1.5px; font-weight: 600; margin-top: 2px; }
    .top-nav-time { color: #94A3B8 !important; font-size: 0.85em; font-weight: 400; }

    /* ===== TYPOGRAPHY ===== */
    h1 { color: #1B2A4A !important; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
         font-weight: 700 !important; letter-spacing: -0.5px; font-size: 1.8em !important; }
    h2 { color: #1B2A4A !important; font-family: -apple-system, sans-serif !important; font-weight: 600 !important;
         font-size: 1.2em !important; border-bottom: 1.5px solid #E2E8F0; padding-bottom: 10px;
         margin-top: 32px !important; margin-bottom: 16px !important; }
    h3 { color: #2E5090 !important; font-family: -apple-system, sans-serif !important; font-weight: 600 !important;
         font-size: 1em !important; margin-top: 20px !important; }
    h4 { color: #1B2A4A !important; font-weight: 600 !important; font-size: 0.95em !important; }
    p, li, span, div, td, th { color: #374151 !important; }

    /* ===== METRICS ===== */
    [data-testid="stMetricValue"] { color: #1B2A4A !important; font-family: -apple-system, sans-serif !important;
        font-weight: 700 !important; font-size: 1.4em !important; }
    [data-testid="stMetricLabel"] { color: #6B7280 !important; text-transform: uppercase !important;
        font-size: 0.6em !important; letter-spacing: 1px !important; font-weight: 600 !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8em !important; }
    [data-testid="stMetricDelta"] span { font-size: 1em !important; }

    /* ===== EXPANDER CARDS ===== */
    .stExpander { border: 1px solid #E5E7EB !important; border-radius: 8px !important;
        background: #FFFFFF !important; margin-bottom: 10px !important;
        transition: all 0.2s ease !important; }
    .stExpander:hover { border-color: #2E5090 !important; box-shadow: 0 2px 12px rgba(46,80,144,0.08) !important;
        transform: translateY(-1px); }
    .stExpander summary, .stExpander [data-testid="stExpanderToggleIcon"] { background: #FFFFFF !important; }
    .stExpander summary span, .stExpander summary p, .stExpander summary div { color: #1B2A4A !important; }
    details[open] summary { border-bottom: 1px solid #E5E7EB; }

    /* ===== FADE-IN ANIMATION ===== */
    .stTabs [data-baseweb="tab-panel"] { animation: fadeIn 0.3s ease-in; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    /* ===== LIVE PULSE INDICATOR ===== */
    .live-dot { width: 8px; height: 8px; background: #10B981; border-radius: 50%; display: inline-block;
        margin-right: 6px; animation: pulse 2s ease-in-out infinite; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

    /* ===== MAIN SECTION TABS (top-level nav) ===== */
    [data-baseweb="tab-list"] {
        background: #FFFFFF !important; border-bottom: 2px solid #E5E7EB; padding: 0; gap: 0; }
    [data-baseweb="tab"] { color: #6B7280 !important; font-weight: 500 !important; font-size: 1.05em !important;
        padding: 14px 28px !important; border-bottom: 3px solid transparent !important;
        background: transparent !important; }
    [data-baseweb="tab"]:hover { color: #1B2A4A !important; }
    [data-baseweb="tab"][aria-selected="true"] { color: #1B2A4A !important; font-weight: 700 !important;
        border-bottom: 3px solid #2E5090 !important; background: transparent !important; }
    [data-baseweb="tab-highlight"] { display: none !important; }
    [data-baseweb="tab-border"] { display: none !important; }

    /* ===== PROGRESS BARS ===== */
    .stProgress > div > div > div { background-color: #2E5090 !important; }
    .stProgress p { color: #374151 !important; font-size: 0.85em !important; font-weight: 500 !important; }

    /* ===== DATAFRAMES ===== */
    .stDataFrame { border: 1px solid #E5E7EB; border-radius: 8px; }

    /* ===== BUTTONS — MAIN CONTENT ===== */
    .stMainBlockContainer .stButton > button {
        background-color: #2E5090 !important; border: none !important; border-radius: 6px !important; }
    .stMainBlockContainer .stButton > button p,
    .stMainBlockContainer .stButton > button span { color: #FFFFFF !important; font-weight: 500 !important; }
    .stMainBlockContainer .stButton > button:hover { background-color: #1B2A4A !important; }
    .stMainBlockContainer .stDownloadButton > button {
        background-color: #FFFFFF !important; border: 1.5px solid #2E5090 !important; border-radius: 6px !important; }
    .stMainBlockContainer .stDownloadButton > button p,
    .stMainBlockContainer .stDownloadButton > button span { color: #2E5090 !important; }
    .stMainBlockContainer .stDownloadButton > button:hover { background-color: #EEF2FF !important; }

    /* ===== CHECKBOX ===== */
    .stCheckbox label span { color: #374151 !important; }

    /* ===== CUSTOM ELEMENTS ===== */
    .header-bar { background: #F0F4F8; padding: 14px 20px; border-radius: 8px; border-left: 3px solid #1B2A4A;
        margin-bottom: 28px; line-height: 1.6; }
    .header-bar, .header-bar * { color: #64748B !important; font-size: 0.88em; }
    .header-bar strong { color: #1B2A4A !important; }
    .thesis-box { background: #F0F4F8; border-left: 3px solid #1B2A4A; padding: 16px 20px; border-radius: 4px;
        margin: 16px 0; line-height: 1.6; }
    .thesis-box, .thesis-box * { color: #1B2A4A !important; font-weight: 500 !important; font-size: 0.93em; }
    .source-cite, .source-cite * { color: #9CA3AF !important; font-size: 0.8em !important; font-style: italic !important; }
    .confidential { color: #DC2626 !important; font-size: 0.65em; text-transform: uppercase; letter-spacing: 2px; font-weight: 700; }
    .accent-line { border-top: 2px solid #2E5090; width: 80px; margin: 6px 0 20px 0; }
    .domain-tag { background: #EEF2FF; padding: 4px 12px; border-radius: 20px;
        font-size: 0.8em; display: inline-block; margin: 2px 4px 2px 0; border: 1px solid #C7D2FE; }
    .domain-tag, .domain-tag * { color: #2E5090 !important; font-weight: 500 !important; }

    /* ===== INFO/WARNING/SUCCESS BOXES ===== */
    .stAlert p, .stAlert span, .stAlert div { color: #374151 !important; }

    /* ===== HIDE STREAMLIT CHROME ===== */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Plotly theme matching PDF style
# ---------------------------------------------------------------------------
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#F8FAFC",
    font=dict(family="-apple-system, BlinkMacSystemFont, sans-serif", color="#374151", size=13),
    xaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#D1D5DB",
               tickfont=dict(size=11, color="#6B7280")),
    yaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB", zerolinecolor="#D1D5DB",
               tickfont=dict(size=11, color="#6B7280")),
    margin=dict(l=60, r=30, t=40, b=60),
    legend=dict(orientation="h", y=1.1, bgcolor="rgba(0,0,0,0)", font=dict(size=12, color="#374151")),
    hoverlabel=dict(bgcolor="#1B2A4A", font_color="white", font_size=12),
)

NAVY = "#1B2A4A"
ACCENT = "#2E5090"
GREEN = "#059669"
RED = "#DC2626"
LIGHT_BG = "#F5F6F8"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json(value):
    if value is None: return []
    if isinstance(value, (list, dict)): return value
    try: return json.loads(value)
    except: return []

def days_held(entry_date):
    try: return (datetime.now() - datetime.strptime(entry_date, "%Y-%m-%d")).days
    except: return 0

def score_emoji(score):
    if score >= 90: return "\U0001f451"
    if score >= 75: return "\U0001f48e"
    if score >= 60: return "\U0001f525"
    if score >= 40: return "\U0001f7e1"
    return "\U0001f4a4"

def fetch_news(ticker, count=3):
    try:
        t = yf.Ticker(ticker)
        news = t.news[:count] if t.news else []
        return [{"title": n.get("title", ""), "link": n.get("link", ""), "publisher": n.get("publisher", "")} for n in news]
    except: return []

def calculate_sharpe(positions, risk_free=0.04):
    returns = [p["pnl_pct"] / 100 for p in positions if p.get("pnl_pct")]
    if len(returns) < 2: return 0.0
    avg = statistics.mean(returns)
    std = statistics.stdev(returns)
    if std == 0: return 0.0
    return round((avg * 12 - risk_free) / (std * (12 ** 0.5)), 2)

def domain_tags_html(domains_str):
    if not domains_str: return ""
    tags = [f'<span class="domain-tag">{d.strip()}</span>' for d in domains_str.split(",") if d.strip()]
    return " ".join(tags)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
stats = get_v2_dashboard_stats()

# Top navigation bar
st.markdown(f'''
<div class="top-nav">
    <div class="top-nav-inner">
        <div class="top-nav-brand">
            <div>
                <div class="top-nav-brand-text">\U0001f48e HUNTER</div>
                <div class="top-nav-brand-sub">Cross-Domain Intelligence</div>
            </div>
        </div>
        <div class="top-nav-stats">
            <div class="top-nav-stat">
                <div class="top-nav-stat-value">{stats["total_facts"]:,}</div>
                <div class="top-nav-stat-label">Facts</div>
            </div>
            <div class="top-nav-stat">
                <div class="top-nav-stat-value">{stats["survived_hypotheses"]}</div>
                <div class="top-nav-stat-label">Diamonds</div>
            </div>
            <div class="top-nav-stat">
                <div class="top-nav-stat-value">{stats["best_score"]}</div>
                <div class="top-nav-stat-label">Best Score</div>
            </div>
            <div class="top-nav-time"><span class="live-dot"></span>{datetime.now().strftime("%d %b %Y %H:%M")}</div>
        </div>
    </div>
</div>
''', unsafe_allow_html=True)

st.markdown("")

# Section navigation as clean tabs at the top
section_tab1, section_tab2, section_tab3, section_tab4, section_tab5, section_tab6 = st.tabs([
    "\U0001f9e0 Intelligence", "\U0001f3ed Research Factory", "\U0001f4b0 Portfolio", "\U0001f3af Targeting", "\U0001f916 HUNTER Sr", "\U0001f423 HUNTER Jr"
])



# ===========================================================================
# SECTION 1: INTELLIGENCE
# ===========================================================================
with section_tab1:
    st.markdown("# Intelligence")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="header-bar">Pipeline Status: <strong>{stats["total_facts"]:,}</strong> facts across '
        f'<strong>18</strong> source types | <strong>{stats["survived_hypotheses"]}</strong> surviving hypotheses | '
        f'<strong>{stats["facts_today"]}</strong> ingested today</div>',
        unsafe_allow_html=True,
    )

    int_tab1, int_tab2, int_tab3, int_tab4 = st.tabs([
        "Source Distribution", "Hypotheses", "Held Queue", "Fact Feed"
    ])

    with int_tab1:
        st.markdown("## Source Distribution")
        diversity = get_source_type_diversity_score()
        if diversity:
            # Build clean horizontal bar chart for all source types
            src_data = []
            for src, pct in diversity.items():
                if src == "test": continue
                icon = SOURCE_ICONS.get(src, "")
                count = int(pct * stats["total_facts"])
                src_type = "Core" if src in CORE_SOURCE_TYPES else "Original"
                src_data.append({"Source": f"{icon} {src}", "Facts": count, "Type": src_type})

            df_src = pd.DataFrame(src_data).sort_values("Facts", ascending=True)
            fig_src = go.Figure()
            for stype, color in [("Original", ACCENT), ("Core", "#059669")]:
                mask = df_src["Type"] == stype
                fig_src.add_trace(go.Bar(
                    y=df_src[mask]["Source"], x=df_src[mask]["Facts"], orientation="h",
                    name=stype, marker_color=color,
                    text=df_src[mask]["Facts"].apply(lambda x: f"{x:,}"), textposition="outside",
                    textfont=dict(size=11, color="#374151"),
                ))
            fig_src.update_layout(**PLOTLY_LAYOUT, height=max(500, len(src_data) * 32),
                                 barmode="group", showlegend=True, yaxis_title="", xaxis_title="Fact Count")
            st.plotly_chart(fig_src, use_container_width=True)

    with int_tab2:
        st.markdown("## Diamond Hypotheses")
        surviving = get_hypotheses_list(min_score=0, survived_only=True, limit=50)
        surviving = [h for h in surviving if h.get("diamond_score")]

        if not surviving:
            st.info("No surviving hypotheses yet. Let HUNTER run.")
        else:
            new_count = sum(1 for h in surviving if h.get("created_at", "") >= (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d"))
            reviewed_count = sum(1 for h in surviving if h.get("reviewed"))
            st.caption(f"{len(surviving)} surviving hypotheses | {new_count} new (24h) | {reviewed_count} reviewed")
            for h in surviving:
                score = h.get("diamond_score", 0) or 0
                emoji = score_emoji(score)
                is_reviewed = h.get("reviewed", 0)
                created = h.get("created_at", "") or ""
                is_new = created >= (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d")

                rev_tag = " \u2705" if is_reviewed else ""
                new_tag = " \U0001f535" if is_new and not is_reviewed else ""

                with st.expander(f"{emoji}{new_tag}{rev_tag}  **{score}/100**  --  {h['hypothesis_text'][:90]}", expanded=False):
                    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
                    with sc1: st.metric("Novelty", f"{h.get('novelty', 0)}/20")
                    with sc2: st.metric("Feasibility", f"{h.get('feasibility', 0)}/20")
                    with sc3: st.metric("Timing", f"{h.get('timing', 0)}/20")
                    with sc4: st.metric("Asymmetry", f"{h.get('asymmetry', 0)}/20")
                    with sc5: st.metric("Intersection", f"{h.get('intersection', 0)}/20")

                    st.markdown(f'<div class="thesis-box">{h["hypothesis_text"]}</div>', unsafe_allow_html=True)

                    chain = _parse_json(h.get("fact_chain"))
                    if chain:
                        st.markdown("#### Fact Chain")
                        for link in chain:
                            if isinstance(link, dict):
                                st.markdown(f"- **Fact #{link.get('fact_id', '?')}**: {link.get('role', '')}")

                    if h.get("action_steps"):
                        st.markdown("#### Action Steps")
                        st.markdown(h["action_steps"])

                    kills = _parse_json(h.get("kill_attempts"))
                    if kills:
                        survived_count = sum(1 for k in kills if not k.get("killed"))
                        st.markdown(f"#### Kill Attempts ({survived_count} survived)")
                        for k in kills:
                            icon_k = "\u2705" if not k.get("killed") else "\u274c"
                            st.markdown(f"- {icon_k} {k.get('round', '?')}: {k.get('reason', '')[:200]}")

                    if h.get("full_report"):
                        with st.expander("Full Report", expanded=False):
                            st.markdown(h["full_report"])

                    new_rev = st.checkbox("Reviewed", value=bool(is_reviewed), key=f"hyp_{h['id']}")
                    if new_rev != bool(is_reviewed):
                        toggle_reviewed("hypotheses", h["id"], new_rev)
                        st.rerun()

    with int_tab3:
        st.markdown("## Held Queue")
        st.caption("Collisions that passed evaluation but failed the search gate. Your manual review queue.")
        held = get_held_collisions_list(limit=50)
        if not held:
            st.info("No held collisions yet.")
        else:
            for h in held:
                is_rev = h.get("reviewed", 0)
                rev_tag = " \u2705" if is_rev else ""
                with st.expander(f"{rev_tag} {h['collision_description'][:90]}", expanded=False):
                    st.markdown(f"**Collision:** {h['collision_description']}")
                    st.markdown(f'<span class="source-cite">Source types: {h.get("source_types", "")}</span>', unsafe_allow_html=True)
                    st.markdown(f'<span class="source-cite">Gate: {h.get("gate_reasoning", "")}</span>', unsafe_allow_html=True)
                    new_rev = st.checkbox("Reviewed", value=bool(is_rev), key=f"held_{h['id']}")
                    if new_rev != bool(is_rev):
                        toggle_reviewed("held_collisions", h["id"], new_rev)
                        st.rerun()

    with int_tab4:
        st.markdown("## Fact Feed")
        facts = get_raw_facts_stream(limit=30)
        if facts:
            for f in facts:
                icon = SOURCE_ICONS.get(f["source_type"], "\U0001f4cc")
                core_b = " \U0001f3af" if f["source_type"] in CORE_SOURCE_TYPES else ""
                with st.expander(f"{icon} {f['title'][:80]}{core_b}", expanded=False):
                    st.markdown(f"**{f['source_type']}** | {f.get('date_of_fact', '?')} | Obscurity: {f.get('obscurity', '?')}")
                    imps = _parse_json(f.get("implications"))
                    if imps:
                        for imp in imps[:2]:
                            st.markdown(f"- _{imp[:180]}_")


# ===========================================================================
# SECTION 2: RESEARCH FACTORY
# ===========================================================================
with section_tab2:
    st.markdown("# Research Factory")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.caption("Enrich diamond theses into professional investment memos")

    all_hyps = get_hypotheses_list(min_score=50, survived_only=True, limit=50)
    all_hyps = [h for h in all_hyps if h.get("diamond_score")]

    if not all_hyps:
        st.info("No diamond hypotheses to enrich yet.")
    else:
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        os.makedirs(reports_dir, exist_ok=True)
        existing_reports = set(os.listdir(reports_dir))

        st.markdown(f'<div class="header-bar"><strong>{len(all_hyps)}</strong> diamond theses available for enrichment</div>', unsafe_allow_html=True)

        if st.button("\U0001f680 Enrich All Diamonds", use_container_width=True):
            with st.spinner("Enriching all theses..."):
                result = subprocess.run(["python", "enrich_thesis.py", "all"], capture_output=True, text=True, timeout=600, cwd="/Users/johnmalpass/HUNTER")
                st.success("Batch enrichment complete!")
                st.code(result.stdout)
                st.rerun()

        st.markdown("---")

        for h in all_hyps:
            score = h.get("diamond_score", 0) or 0
            emoji = score_emoji(score)
            hyp_id = h["id"]
            report_name = f"HUNTER_Thesis_{hyp_id}.pdf"
            has_report = report_name in existing_reports

            pdf_badge = "  \U0001f4c4 PDF Ready" if has_report else ""
            hyp_title = h['hypothesis_text'][:80]
            with st.expander(f"{emoji}  **{score}/100**{pdf_badge}  --  {hyp_title}", expanded=False):
                st.markdown(f'<div class="thesis-box">{h["hypothesis_text"]}</div>', unsafe_allow_html=True)
                st.markdown(f"**Score:** {score} | **Confidence:** {h.get('confidence', '?')} | **Window:** {h.get('time_window_days', '?')} days")

                if has_report:
                    report_path = os.path.join(reports_dir, report_name)
                    with open(report_path, "rb") as f:
                        st.download_button("\U0001f4e5 Download PDF", data=f.read(), file_name=report_name, mime="application/pdf", key=f"dl_{hyp_id}")
                else:
                    if st.button(f"Enrich to PDF", key=f"enrich_{hyp_id}"):
                        with st.spinner("Enriching..."):
                            try:
                                result = subprocess.run(["python", "enrich_thesis.py", str(hyp_id)], capture_output=True, text=True, timeout=120, cwd="/Users/johnmalpass/HUNTER")
                                if result.returncode == 0:
                                    st.success("PDF generated!")
                                    st.rerun()
                                else:
                                    st.error(f"Error: {result.stderr[:200]}")
                            except Exception as e:
                                st.error(f"Failed: {e}")

                # Top 5 firms targeting
                st.markdown("---")
                existing_firms = get_firm_suggestions(hyp_id)
                if existing_firms:
                    st.markdown("**Target Firms:**")
                    for firm in existing_firms:
                        if isinstance(firm, dict):
                            st.markdown(f"- **{firm.get('name', '?')}** ({firm.get('type', '?')}) - _{firm.get('why', '')}_")
                else:
                    if st.button("Find Target Firms", key=f"firms_{hyp_id}"):
                        with st.spinner("Identifying target firms..."):
                            try:
                                firms = suggest_firms(h.get("hypothesis_text", ""), score, "", h.get("direction", "long"))
                                if firms:
                                    save_firm_suggestions(hyp_id, firms)
                                    st.rerun()
                                else:
                                    st.caption("Could not identify target firms.")
                            except Exception as e:
                                st.error(f"Failed: {e}")


# ===========================================================================
# SECTION 3: PORTFOLIO
# ===========================================================================
with section_tab3:
    st.markdown("# Portfolio")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)

    p_stats = get_portfolio_stats()
    p_open = get_open_positions()
    p_closed = get_closed_positions()
    p_all = get_all_positions()
    p_snapshots = get_portfolio_snapshots()

    spy_ret = p_snapshots[0]["spy_return_pct"] if p_snapshots else 0.0
    alpha = p_stats["total_return_pct"] - spy_ret
    sharpe = calculate_sharpe(p_closed)

    st.markdown(
        f'<div class="header-bar">'
        f'<span class="confidential">CONFIDENTIAL</span> | '
        f'Paper Trading Portfolio | Last updated: {datetime.now().strftime("%d %b %Y %H:%M")} | '
        f'{p_stats["num_open"]} open, {p_stats["num_closed"]} closed</div>',
        unsafe_allow_html=True,
    )

    pt1, pt2, pt3, pt4, pt5, pt6, pt7 = st.tabs([
        "Overview", "Open Positions", "Closed", "Risk", "Analytics", "Calibration", "Export"
    ])

    # --- Overview ---
    with pt1:
        # Hero section — Yahoo Finance style
        total_val = p_stats['total_value']
        total_ret = p_stats['total_return_pct']
        total_pnl = p_stats['total_pnl']
        ret_color = GREEN if total_ret >= 0 else RED

        st.markdown(f"""
        <div style="padding: 24px 0 16px 0;">
            <div style="color: #6B7280; font-size: 0.85em; font-weight: 500; text-transform: uppercase; letter-spacing: 1px;">HUNTER Portfolio</div>
            <div style="display: flex; align-items: baseline; gap: 16px; margin-top: 4px;">
                <span style="font-size: 2.8em; font-weight: 700; color: #1B2A4A; font-family: -apple-system, sans-serif;">GBP {total_val:,.0f}</span>
                <span style="font-size: 1.2em; font-weight: 600; color: {ret_color};">{total_pnl:+,.0f} ({total_ret:+.2f}%)</span>
            </div>
            <div style="color: #9CA3AF; font-size: 0.8em; margin-top: 4px;">
                <span class="live-dot"></span>As of {datetime.now().strftime("%d %b %Y %H:%M")} | Paper Trading | Inception: GBP 1,000,000
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Equity chart — shows P&L in GBP on deployed capital
        if len(p_snapshots) >= 2:
            df = pd.DataFrame(p_snapshots).sort_values("date")

            # Calculate P&L in GBP (total_value minus inception)
            df["pnl_gbp"] = df["total_value"] - 1_000_000
            df["spy_pnl"] = 1_000_000 * (df["spy_return_pct"] / 100)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["pnl_gbp"], name="HUNTER P&L",
                line=dict(color=ACCENT, width=3),
                fill="tozeroy", fillcolor="rgba(46,80,144,0.08)",
                hovertemplate="<b>HUNTER</b><br>%{x}<br>GBP %{y:+,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=df["date"], y=df["spy_pnl"], name="S&P 500 (equivalent)",
                line=dict(color="#9CA3AF", width=1.5, dash="dot"),
                hovertemplate="<b>S&P 500</b><br>%{x}<br>GBP %{y:+,.0f}<extra></extra>",
            ))
            # Zero line (breakeven)
            fig.add_hline(y=0, line_dash="solid", line_color="#D1D5DB", line_width=1,
                         annotation_text="Breakeven", annotation_position="bottom left",
                         annotation_font_color="#9CA3AF", annotation_font_size=10)

            chart_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "hoverlabel")}
            fig.update_layout(
                **chart_layout, height=480,
                xaxis=dict(gridcolor="#F0F0F0", linecolor="#E5E7EB", showgrid=False,
                          tickfont=dict(size=11, color="#9CA3AF")),
                yaxis=dict(gridcolor="#F0F0F0", linecolor="#E5E7EB",
                          tickfont=dict(size=11, color="#9CA3AF"), tickprefix="GBP ", side="right",
                          zeroline=True, zerolinecolor="#E5E7EB"),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#1B2A4A", font_color="white", font_size=12, bordercolor="#2E5090"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 12px; padding: 60px 40px; text-align: center; margin: 20px 0;">
                <div style="font-size: 2em; margin-bottom: 12px;">📈</div>
                <div style="color: #6B7280; font-size: 1em;">Run <code>python portfolio.py update</code> daily to build the equity curve</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        # Stats row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1: st.metric("SPY Benchmark", f"{spy_ret:+.2f}%")
        with c2: st.metric("Alpha", f"{alpha:+.2f}%")
        with c3: st.metric("Win Rate", f"{p_stats['win_rate']:.0f}%")
        with c4: st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        with c5: st.metric("Open Positions", f"{p_stats['num_open']}")
        with c6: st.metric("Closed", f"{p_stats['num_closed']}")

        # Positions table with HUNTER scores
        if p_open:
            st.markdown("## Active Positions")
            pos_rows = []
            for p in p_open:
                pnl = p.get("pnl_pct", 0) or 0
                score = p.get("diamond_score", 0) or 0
                emoji = score_emoji(score)
                arrow = "\u2191" if p["direction"] == "long" else "\u2193"
                pos_rows.append({
                    "": f"{emoji}",
                    "Ticker": p["ticker"],
                    "Direction": f"{arrow} {p['direction'].upper()}",
                    "HUNTER Score": f"{score}/100",
                    "Entry": f"${p['entry_price']:.2f}" if p.get("entry_price") else "-",
                    "Current": f"${p['current_price']:.2f}" if p.get("current_price") else "-",
                    "P&L": f"{pnl:+.2f}%",
                    "Capital": f"GBP {p.get('capital_allocated', 0):,.0f}",
                    "Days Held": days_held(p.get("entry_date", "")),
                })
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

            # Capital allocation pie
            st.markdown("## Capital Allocation")
            total_alloc = sum(p.get("capital_allocated", 0) or 0 for p in p_open)
            pie_data = [{"Ticker": f"{p['ticker']} ({p.get('diamond_score', '?')})", "Capital": p.get("capital_allocated", 0) or 0} for p in p_open]
            df_pie = pd.DataFrame(pie_data)

            fig_pie = px.pie(df_pie, values="Capital", names="Ticker", hole=0.5,
                           color_discrete_sequence=[ACCENT, NAVY, GREEN, "#F59E0B", "#8B5CF6", "#EC4899", "#6B7280", "#D0D3D9", "#0EA5E9", "#14B8A6"])
            fig_pie.update_layout(**PLOTLY_LAYOUT, height=420)
            fig_pie.update_traces(textinfo="label+percent", textfont_size=11,
                                 hovertemplate="<b>%{label}</b><br>GBP %{value:,.0f}<br>%{percent}<extra></extra>")
            st.plotly_chart(fig_pie, use_container_width=True)

            st.caption(f"Deployed: GBP {total_alloc:,.0f} of GBP 1,000,000 ({total_alloc/1_000_000*100:.1f}%) | Cash: GBP {1_000_000 - total_alloc:,.0f}")

    # --- Open Positions ---
    with pt2:
        st.markdown("## Open Positions")
        if not p_open:
            st.info("No open positions. Run `python portfolio.py log` to create positions.")
        else:
            for p in p_open:
                pnl = p.get("pnl_pct", 0) or 0
                pnl_gbp = p.get("pnl_gbp", 0) or 0
                pnl_class = "pnl-positive" if pnl >= 0 else "pnl-negative"
                arrow = "\u2191" if p["direction"] == "long" else "\u2193"
                held = days_held(p.get("entry_date", ""))
                window = p.get("time_window_days", 90) or 90
                entry_str = f"${p['entry_price']:.2f}" if p.get("entry_price") else "MANUAL"
                current_str = f"${p['current_price']:.2f}" if p.get("current_price") else "-"

                score = p.get('diamond_score', 0) or 0
                emoji = score_emoji(score)

                with st.expander(
                    f"{emoji} **{p['ticker']}** {arrow}  |  {entry_str} -> {current_str}  |  "
                    f"**{pnl:+.2f}%**  |  Day {held}/{window}  |  HUNTER Score: {score}/100",
                    expanded=False,
                ):
                    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
                    with mc1: st.metric("HUNTER Score", f"{score}/100")
                    with mc2: st.metric("Entry Price", entry_str)
                    with mc3: st.metric("Current Price", current_str)
                    with mc4: st.metric("P&L", f"{pnl:+.2f}%", delta=f"GBP {pnl_gbp:+,.0f}")
                    with mc5: st.metric("Capital", f"GBP {p.get('capital_allocated', 0):,.0f}")

                    # Time progress
                    progress = min(held / window, 1.0) if window > 0 else 0
                    st.progress(progress, text=f"Day {held} of {window} ({progress*100:.0f}% elapsed)")

                    # Thesis
                    st.markdown(f'<div class="thesis-box">{(p.get("hypothesis_text", "") or "")[:500]}</div>', unsafe_allow_html=True)

                    # Domains
                    if p.get("domains"):
                        st.markdown(domain_tags_html(p["domains"]), unsafe_allow_html=True)

                    # News
                    news = fetch_news(p["ticker"])
                    st.markdown("**Recent Headlines:**")
                    valid_news = [n for n in (news or []) if n.get("title") and n.get("link")]
                    if valid_news:
                        for n in valid_news:
                            st.markdown(f'- [{n["title"][:80]}]({n["link"]}) <span class="source-cite">{n["publisher"]}</span>', unsafe_allow_html=True)
                    else:
                        st.caption("No recent headlines available for this ticker.")

    # --- Closed ---
    with pt3:
        st.markdown("## Closed Positions")
        if not p_closed:
            st.info("No closed positions yet. Positions auto-close when their time window expires.")
        else:
            st.markdown(
                f'<div class="header-bar">'
                f'Win rate: <strong>{p_stats["win_rate"]:.0f}%</strong> | '
                f'Avg return: <strong>{p_stats["avg_return"]:+.2f}%</strong> | '
                f'Best: <strong>{p_stats["best_trade"]:+.2f}%</strong> | '
                f'Worst: <strong>{p_stats["worst_trade"]:+.2f}%</strong></div>',
                unsafe_allow_html=True,
            )
            for p in p_closed:
                pnl = p.get("pnl_pct", 0) or 0
                icon = "\u2705" if pnl > 0 else "\u274c"
                with st.expander(f"{icon} **{p['ticker']}** {p['direction']} | **{pnl:+.2f}%** (GBP {p.get('pnl_gbp', 0):+,.0f}) | {p.get('close_reason', '?')}", expanded=False):
                    st.markdown(f'<div class="thesis-box">{(p.get("hypothesis_text", "") or "")[:300]}</div>', unsafe_allow_html=True)
                    st.markdown(f"**Opened:** {p.get('entry_date', '?')} | **Closed:** {p.get('close_date', '?')} | **Reason:** {p.get('close_reason', '?')}")

    # --- Risk ---
    with pt4:
        st.markdown("## Risk Analysis")
        if p_open:
            total_deployed = sum(p.get("capital_allocated", 0) or 0 for p in p_open)
            long_cap = sum(p.get("capital_allocated", 0) or 0 for p in p_open if p["direction"] == "long")
            short_cap = sum(p.get("capital_allocated", 0) or 0 for p in p_open if p["direction"] == "short")

            rc1, rc2, rc3, rc4 = st.columns(4)
            with rc1: st.metric("Long Exposure", f"GBP {long_cap:,.0f}")
            with rc2: st.metric("Short Exposure", f"GBP {short_cap:,.0f}")
            with rc3: st.metric("Net Exposure", f"GBP {long_cap - short_cap:+,.0f}")
            with rc4: st.metric("Cash Reserve", f"GBP {1_000_000 - total_deployed:,.0f}")

            # Domain concentration
            st.markdown("### Domain Concentration")
            domain_cap = {}
            for p in p_open:
                for d in (p.get("domains") or "Unknown").split(","):
                    d = d.strip()
                    if d:
                        domain_cap[d] = domain_cap.get(d, 0) + (p.get("capital_allocated", 0) or 0) / max(len((p.get("domains") or "").split(",")), 1)

            if domain_cap and total_deployed > 0:
                df_c = pd.DataFrame({"Domain": list(domain_cap.keys()), "Capital %": [round(v/total_deployed*100, 1) for v in domain_cap.values()]})
                df_c = df_c.sort_values("Capital %", ascending=True)

                # Limit to top 10 domains for readability
                df_c = df_c.tail(10)
                fig_conc = go.Figure(go.Bar(
                    x=df_c["Capital %"], y=df_c["Domain"], orientation="h",
                    marker_color=[RED if v > 25 else ACCENT for v in df_c["Capital %"]],
                    text=[f"{v:.0f}%" for v in df_c["Capital %"]], textposition="outside",
                    textfont=dict(size=11, color="#374151"),
                ))
                fig_conc.update_layout(**PLOTLY_LAYOUT, height=max(300, len(df_c) * 38 + 80),
                                      xaxis_title="% of Deployed Capital", yaxis_title="")
                st.plotly_chart(fig_conc, use_container_width=True)

                for d, cap in domain_cap.items():
                    pct = cap / total_deployed * 100
                    if pct > 25:
                        st.warning(f"**{d}** has {pct:.0f}% concentration - consider diversifying")

            # Correlation flags
            st.markdown("### Correlation Flags")
            tickers_by_domain = {}
            for p in p_open:
                for d in (p.get("domains") or "").split(","):
                    d = d.strip()
                    if d: tickers_by_domain.setdefault(d, []).append(p["ticker"])

            flagged = False
            for domain, tickers in tickers_by_domain.items():
                if len(tickers) >= 2:
                    st.warning(f"**{domain}**: {', '.join(tickers)} are correlated bets")
                    flagged = True
            if not flagged:
                st.success("No correlated position clusters detected")
        else:
            st.info("No open positions to analyze.")

    # --- Analytics ---
    with pt5:
        st.markdown("## Domain Performance")
        domain_perf = {}
        for p in p_all:
            for d in (p.get("domains") or "Unknown").split(","):
                d = d.strip()
                if d: domain_perf.setdefault(d, []).append(p.get("pnl_pct", 0) or 0)

        if domain_perf:
            rows = [{"Domain": d, "Positions": len(r), "Avg Return %": round(statistics.mean(r), 2)} for d, r in domain_perf.items()]
            df_d = pd.DataFrame(rows).sort_values("Avg Return %", ascending=False).head(12)  # Top 12 only

            # Horizontal bar — much more readable with domain names
            fig_dom = go.Figure(go.Bar(
                y=df_d["Domain"], x=df_d["Avg Return %"], orientation="h",
                marker_color=[GREEN if v >= 0 else RED for v in df_d["Avg Return %"]],
                text=[f"{v:+.1f}%" for v in df_d["Avg Return %"]], textposition="outside",
                textfont=dict(size=11, color="#374151"),
            ))
            fig_dom.update_layout(**PLOTLY_LAYOUT, height=max(350, len(df_d) * 36),
                                 xaxis_title="Avg Return %", yaxis_title="")
            st.plotly_chart(fig_dom, use_container_width=True)

            # Full table below
            st.dataframe(pd.DataFrame(rows).sort_values("Avg Return %", ascending=False),
                        use_container_width=True, hide_index=True)

        # Score vs P&L scatter
        if p_all:
            st.markdown("## Score vs Performance")
            scatter = [{"Score": p.get("diamond_score", 0) or 0, "P&L %": p.get("pnl_pct", 0) or 0,
                        "Ticker": p.get("ticker", "?"), "Direction": p.get("direction", "?")} for p in p_all if p.get("diamond_score")]
            if scatter:
                df_s = pd.DataFrame(scatter)
                fig_scatter = go.Figure()
                for _, row in df_s.iterrows():
                    color = GREEN if row["P&L %"] > 0 else RED if row["P&L %"] < 0 else "#9CA3AF"
                    fig_scatter.add_trace(go.Scatter(
                        x=[row["Score"]], y=[row["P&L %"]], mode="markers+text",
                        text=[row["Ticker"]], textposition="top center",
                        textfont=dict(size=12, color="#1B2A4A", family="-apple-system, sans-serif"),
                        marker=dict(size=18, color=color, line=dict(width=1.5, color="#1B2A4A"), opacity=0.85),
                        hovertemplate=f"<b>{row['Ticker']}</b><br>Score: {row['Score']}<br>P&L: {row['P&L %']:+.2f}%<br>Direction: {row['Direction']}<extra></extra>",
                        showlegend=False,
                    ))
                scatter_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
                fig_scatter.update_layout(**scatter_layout, height=480,
                                        xaxis_title="Diamond Score", yaxis_title="P&L %",
                                        xaxis=dict(range=[45, 105], gridcolor="#E5E7EB", linecolor="#D1D5DB",
                                                   tickfont=dict(size=11, color="#6B7280")),
                                        yaxis=dict(gridcolor="#E5E7EB", linecolor="#D1D5DB",
                                                   tickfont=dict(size=11, color="#6B7280")))
                st.plotly_chart(fig_scatter, use_container_width=True)

    # --- Calibration ---
    with pt6:
        st.markdown("## Confidence Calibration")
        st.caption("Are HUNTER's confidence scores predictive? This is the proof that the system knows something real.")

        # Gather all positions for calibration analysis
        cal_all = p_all

        if not cal_all or len(cal_all) < 3:
            st.markdown("""
            <div style="background: #F8FAFC; border: 1px solid #E5E7EB; border-radius: 12px; padding: 60px 40px; text-align: center; margin: 20px 0;">
                <div style="font-size: 2em; margin-bottom: 12px;">📊</div>
                <div style="color: #1B2A4A; font-size: 1.1em; font-weight: 600; margin-bottom: 8px;">Calibration Data Building</div>
                <div style="color: #6B7280; font-size: 0.9em; line-height: 1.6;">
                    Calibration requires closed positions to measure predicted vs actual outcomes.<br>
                    Currently tracking {len(cal_all)} positions. Meaningful calibration at 20+ closed positions.<br>
                    First statistical significance at ~90 days of runtime.
                </div>
            </div>
            """.format(len(cal_all)), unsafe_allow_html=True)

        # Chart 1: Confidence Calibration Curve
        st.markdown("### Chart 1: Score Band vs Win Rate")
        st.caption("A calibrated system produces a diagonal — higher scores should win more often")

        score_buckets = {"50-59": (50, 59), "60-69": (60, 69), "70-79": (70, 79), "80-89": (80, 89), "90-100": (90, 100)}
        cal_data = []
        for label, (low, high) in score_buckets.items():
            bucket_pos = [p for p in cal_all if low <= (p.get("diamond_score", 0) or 0) <= high]
            if bucket_pos:
                wins = sum(1 for p in bucket_pos if (p.get("pnl_pct", 0) or 0) > 0)
                total_in_bucket = len(bucket_pos)
                avg_pnl = statistics.mean([p.get("pnl_pct", 0) or 0 for p in bucket_pos])
                win_rate = wins / total_in_bucket * 100
                expected_wr = (low + high) / 2  # ideal: score ~= win rate
                cal_data.append({
                    "Score Band": label,
                    "Positions": total_in_bucket,
                    "Win Rate %": round(win_rate, 1),
                    "Expected %": expected_wr,
                    "Avg P&L %": round(avg_pnl, 2),
                    "Wins": wins,
                    "Losses": total_in_bucket - wins,
                })

        if cal_data:
            df_cal = pd.DataFrame(cal_data)

            fig_cal = go.Figure()
            fig_cal.add_trace(go.Bar(
                x=df_cal["Score Band"], y=df_cal["Win Rate %"],
                name="Actual Win Rate", marker_color=ACCENT,
                text=[f"{v:.0f}%" for v in df_cal["Win Rate %"]], textposition="outside",
                textfont=dict(size=12, color="#374151"),
            ))
            fig_cal.add_trace(go.Scatter(
                x=df_cal["Score Band"], y=df_cal["Expected %"],
                name="Perfect Calibration", mode="lines+markers",
                line=dict(color="#F59E0B", dash="dash", width=2),
                marker=dict(size=8, color="#F59E0B"),
            ))
            cal_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "legend"}
            fig_cal.update_layout(**cal_layout, height=400, yaxis_title="Win Rate %",
                                 barmode="group", legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig_cal, use_container_width=True)

            st.dataframe(df_cal, use_container_width=True, hide_index=True)
        else:
            st.info("Need positions with scores to build calibration curve.")

        st.markdown("---")

        # Chart 2: Expected vs Actual P&L by Score Tier
        st.markdown("### Chart 2: Average P&L by Score Tier")
        st.caption("Do higher-score theses win by MORE, not just more often?")

        if cal_data:
            fig_pnl = go.Figure(go.Bar(
                x=[d["Score Band"] for d in cal_data],
                y=[d["Avg P&L %"] for d in cal_data],
                marker_color=[GREEN if d["Avg P&L %"] >= 0 else RED for d in cal_data],
                text=[f"{d['Avg P&L %']:+.1f}%" for d in cal_data], textposition="outside",
                textfont=dict(size=12, color="#374151"),
            ))
            fig_pnl.update_layout(**PLOTLY_LAYOUT, height=350, yaxis_title="Avg P&L %")
            st.plotly_chart(fig_pnl, use_container_width=True)

        st.markdown("---")

        # Chart 3: Score vs Realised Return (scatter with regression)
        st.markdown("### Chart 3: Score vs Realised Return")
        st.caption("Each dot is a position. Upward slope = scores predict returns.")

        scatter_cal = [{"Score": p.get("diamond_score", 0) or 0,
                       "P&L %": p.get("pnl_pct", 0) or 0,
                       "Ticker": p.get("ticker", "?"),
                       "Status": p.get("status", "?")}
                      for p in cal_all if p.get("diamond_score")]

        if scatter_cal and len(scatter_cal) >= 2:
            df_sc = pd.DataFrame(scatter_cal)

            fig_sc = go.Figure()
            for _, row in df_sc.iterrows():
                color = GREEN if row["P&L %"] > 0 else RED if row["P&L %"] < 0 else "#9CA3AF"
                fig_sc.add_trace(go.Scatter(
                    x=[row["Score"]], y=[row["P&L %"]], mode="markers+text",
                    text=[row["Ticker"]], textposition="top center",
                    textfont=dict(size=11, color="#1B2A4A"),
                    marker=dict(size=16, color=color, line=dict(width=1.5, color="#1B2A4A"), opacity=0.85),
                    showlegend=False,
                    hovertemplate=f"<b>{row['Ticker']}</b><br>Score: {row['Score']}<br>P&L: {row['P&L %']:+.2f}%<extra></extra>",
                ))

            # Add regression line if enough data
            if len(df_sc) >= 3:
                import numpy as np
                x = df_sc["Score"].values.astype(float)
                y = df_sc["P&L %"].values.astype(float)
                if x.std() > 0:
                    slope, intercept = np.polyfit(x, y, 1)
                    x_line = np.array([x.min(), x.max()])
                    y_line = slope * x_line + intercept
                    fig_sc.add_trace(go.Scatter(
                        x=x_line, y=y_line, mode="lines", name="Trend",
                        line=dict(color="#F59E0B", dash="dash", width=2),
                    ))
                    slope_label = "positive (scores predict returns)" if slope > 0 else "flat/negative (scores NOT predictive yet)"
                    st.markdown(f'<div class="header-bar">Regression slope: <strong>{slope:.4f}</strong> - {slope_label}</div>', unsafe_allow_html=True)

            sc_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
            fig_sc.update_layout(**sc_layout, height=450,
                                xaxis_title="Diamond Score", yaxis_title="Realised P&L %",
                                xaxis=dict(range=[45, 105], gridcolor="#E5E7EB", tickfont=dict(size=11, color="#6B7280")),
                                yaxis=dict(gridcolor="#E5E7EB", tickfont=dict(size=11, color="#6B7280")))
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("Need more positions to build the scatter plot.")

        st.markdown("---")

        # Summary stats
        st.markdown("### Calibration Summary")
        if cal_all:
            total_pos = len(cal_all)
            closed_count = len(p_closed)
            open_count = len(p_open)
            winners = sum(1 for p in cal_all if (p.get("pnl_pct", 0) or 0) > 0)
            losers = sum(1 for p in cal_all if (p.get("pnl_pct", 0) or 0) < 0)
            flat = total_pos - winners - losers

            sc1, sc2, sc3, sc4 = st.columns(4)
            with sc1: st.metric("Total Positions", f"{total_pos}")
            with sc2: st.metric("Winners", f"{winners}")
            with sc3: st.metric("Losers", f"{losers}")
            with sc4: st.metric("Pending", f"{flat}")

            st.markdown(f"""
            <div class="header-bar">
                <strong>Calibration status:</strong> {'Statistically significant' if closed_count >= 20 else f'Building — {closed_count} closed, need 20+ for significance'}
                <br>First meaningful read at ~90 days of runtime. Statistical significance at ~180 days.
            </div>
            """, unsafe_allow_html=True)

    # --- Export ---
    with pt7:
        st.markdown("## Export")
        st.markdown(f"""
        <div class="header-bar">
        <strong>HUNTER Portfolio Proof Document</strong><br>
        Inception: GBP 1,000,000 | Current: GBP {p_stats['total_value']:,.0f} ({p_stats['total_return_pct']:+.2f}%)<br>
        Positions: {p_stats['num_open']} open, {p_stats['num_closed']} closed | Generated: {datetime.now().strftime('%d %B %Y')}
        </div>
        """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("\U0001f4c4 Generate HTML Report", use_container_width=True):
                result = subprocess.run(["python", "portfolio.py", "report"], capture_output=True, text=True, cwd="/Users/johnmalpass/HUNTER")
                st.success("Report generated at HUNTER_Portfolio_Report.html")
                st.code(result.stdout)
        with col_b:
            if st.button("\U0001f504 Update Prices Now", use_container_width=True):
                result = subprocess.run(["python", "portfolio.py", "update"], capture_output=True, text=True, cwd="/Users/johnmalpass/HUNTER")
                st.success("Prices updated!")
                st.code(result.stdout)
                st.rerun()


# ===========================================================================
# SECTION 4: TARGETING
# ===========================================================================
with section_tab4:
    st.markdown("# Targeting")
    st.markdown('<div class="accent-line"></div>', unsafe_allow_html=True)
    st.caption("Direct HUNTER's intelligence toward specific firms and strategies")

    tgt_tab1, tgt_tab2 = st.tabs([
        "Target Firms", "Domain Generator"
    ])

    # --- Target Firms ---
    with tgt_tab1:
        st.markdown("## Active Targets")
        st.caption("Direct HUNTER's intelligence toward specific firms and their verticals")

        targets = get_active_targets()

        # Show existing targets as rich cards
        if targets:
            for t in targets:
                firm_name = t.get("firm_name", "Unknown")
                verticals = t.get("verticals", "")
                focus_domains = t.get("focus_domains", "")
                notes = t.get("notes", "")
                created = (t.get("created_at", "") or "")[:10]

                # Count facts in this firm's focus domains
                focus_list = [d.strip() for d in focus_domains.split(",") if d.strip()]
                domain_fact_counts = {}
                if focus_list:
                    try:
                        tgt_conn = get_connection()
                        tgt_cursor = tgt_conn.cursor()
                        for fd in focus_list:
                            tgt_cursor.execute("SELECT COUNT(*) as cnt FROM raw_facts WHERE source_type = ?", (fd,))
                            domain_fact_counts[fd] = tgt_cursor.fetchone()["cnt"]
                        tgt_conn.close()
                    except: pass

                total_focus_facts = sum(domain_fact_counts.values())

                # Count hypotheses relevant to this firm's domains
                relevant_hyps = 0
                try:
                    tgt_conn = get_connection()
                    tgt_cursor = tgt_conn.cursor()
                    tgt_cursor.execute("SELECT COUNT(*) as cnt FROM hypotheses WHERE survived_kill = 1 AND diamond_score >= 50")
                    relevant_hyps = tgt_cursor.fetchone()["cnt"]
                    tgt_conn.close()
                except: pass

                with st.expander(f"**{firm_name}** | {len(focus_list)} domains | {total_focus_facts:,} focus facts", expanded=True):
                    # Header
                    st.markdown(f"""
                    <div style="background: #F0F4F8; border-radius: 8px; padding: 20px; border: 1px solid #E2E8F0; margin-bottom: 16px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div>
                                <div style="font-size: 1.3em; font-weight: 700; color: #1B2A4A;">{firm_name}</div>
                                <div style="color: #6B7280; font-size: 0.85em; margin-top: 4px;">Active since {created} | Weight: {t['weight']}</div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.4em; font-weight: 700; color: #2E5090;">{total_focus_facts:,}</div>
                                <div style="color: #6B7280; font-size: 0.65em; text-transform: uppercase; letter-spacing: 1px;">Focus Facts</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Verticals
                    if verticals:
                        st.markdown("**Verticals:**")
                        st.markdown(domain_tags_html(verticals), unsafe_allow_html=True)

                    # Focus domain breakdown
                    if domain_fact_counts:
                        st.markdown("")
                        st.markdown("**Domain Fact Density:**")
                        for fd, cnt in sorted(domain_fact_counts.items(), key=lambda x: -x[1]):
                            icon = SOURCE_ICONS.get(fd, "")
                            st.progress(min(cnt / max(max(domain_fact_counts.values()), 1), 1.0),
                                       text=f"{icon} {fd}: {cnt:,} facts")

                    # Notes
                    if notes:
                        st.markdown("")
                        st.markdown(f'<div class="source-cite">{notes}</div>', unsafe_allow_html=True)

                    # Stats row
                    st.markdown("")
                    ts1, ts2, ts3 = st.columns(3)
                    with ts1: st.metric("Focus Domains", f"{len(focus_list)}")
                    with ts2: st.metric("Focus Facts", f"{total_focus_facts:,}")
                    with ts3: st.metric("Diamonds", f"{relevant_hyps}")

                    # Remove button
                    st.markdown("")
                    if st.button("Remove Target", key=f"rm_tgt_{t['id']}"):
                        remove_target(t["id"])
                        st.rerun()

            st.markdown("---")

        # Add new target
        st.markdown("### Add New Target")
        col_name, col_weight = st.columns([3, 1])
        with col_name:
            new_firm = st.text_input("Firm Name", placeholder="e.g. Apollo Global Management")
        with col_weight:
            new_weight = st.number_input("Weight", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        new_verticals = st.text_input("Verticals", placeholder="e.g. Distressed Credit, Insurance, Real Estate")
        new_focus = st.text_input("Focus Domains", placeholder="e.g. distressed, insurance, cre_credit")
        new_notes = st.text_area("Notes", placeholder="Strategy notes, recent deals, contact info", height=80)

        if st.button("Add Target", use_container_width=True):
            if new_firm:
                save_target(new_firm, new_weight, new_verticals, new_focus, new_notes)
                st.success(f"Added {new_firm}")
                st.rerun()

    # --- Domain Generator ---
    with tgt_tab2:
        st.markdown("## Domain Generator")
        st.caption("AI-suggested new domains based on target firm strategies")

        targets = get_active_targets()
        if not targets:
            st.info("Add target firms first to generate domain suggestions.")
        else:
            selected_firm = st.selectbox("Select firm", [t["firm_name"] for t in targets])

            if st.button("Generate Domain Suggestions", use_container_width=True):
                with st.spinner(f"Analysing {selected_firm}'s strategy..."):
                    try:
                        domains = generate_domains_for_firm(selected_firm)
                        if domains:
                            for d in domains:
                                if isinstance(d, dict):
                                    st.markdown(f"### {d.get('domain', 'Unknown')}")
                                    st.markdown(f"_{d.get('description', '')}_")
                                    if d.get("example_queries"):
                                        st.markdown("**Example queries:**")
                                        for q in d["example_queries"]:
                                            st.markdown(f"- `{q}`")
                                    st.markdown("---")
                        else:
                            st.warning("No domain suggestions generated.")
                    except Exception as e:
                        st.error(f"Failed: {e}")


# ===========================================================================
# SECTION 5: HUNTER Sr (AI Overseer)
# ===========================================================================
with section_tab5:
    st.markdown("")
    st.markdown("")

    # Big robot face header
    sr_col1, sr_col2, sr_col3 = st.columns([1, 2, 1])
    with sr_col2:
        st.markdown("""
        <div style="text-align: center; padding: 30px 0 20px 0;">
            <div style="font-size: 5em; line-height: 1;">🤖</div>
            <div style="font-size: 1.6em; font-weight: 700; color: #1B2A4A; margin-top: 8px;">HUNTER Sr</div>
            <div style="color: #6B7280; font-size: 0.95em; margin-top: 4px;">System Intelligence & Performance Analyst</div>
            <div style="border-top: 2px solid #2E5090; width: 60px; margin: 16px auto 0 auto;"></div>
        </div>
        """, unsafe_allow_html=True)

    # Latest report
    latest = get_latest_overseer_report()

    if latest:
        st.markdown(f"""
        <div class="header-bar" style="text-align: center;">
            Last analysis: <strong>{(latest.get("created_at", "") or "")[:16]}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Metrics from report
        try:
            metrics = json.loads(latest.get("metrics_json", "{}") or "{}")
            if metrics:
                m1, m2, m3, m4, m5 = st.columns(5)
                with m1: st.metric("Facts", f"{metrics.get('total_facts', 0):,}")
                with m2: st.metric("Diamonds", f"{metrics.get('survived_hypotheses', 0)}")
                with m3: st.metric("Best Score", f"{metrics.get('best_score', 0)}")
                with m4: st.metric("Portfolio", f"GBP {metrics.get('portfolio_value', 100000):,.0f}")
                with m5: st.metric("Win Rate", f"{metrics.get('win_rate', 0):.0f}%")
        except: pass

        st.markdown("---")

        # Report text
        st.markdown("## Analysis Report")
        st.markdown(latest.get("report_text", "No report available."))

        # Suggestions
        suggestions = []
        try:
            suggestions = json.loads(latest.get("suggestions_json", "[]") or "[]")
        except: pass

        if suggestions:
            st.markdown("## Recommendations")
            for i, s in enumerate(suggestions[:7], 1):
                st.markdown(f"""
                <div style="background: #F0F4F8; border-left: 3px solid #2E5090; padding: 12px 16px;
                     border-radius: 4px; margin: 8px 0;">
                    <strong style="color: #2E5090;">{i}.</strong> {s}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 0; color: #9CA3AF;">
            <p style="font-size: 1.1em;">No analysis run yet. Click below to let HUNTER Sr analyse the system.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    if st.button("🤖 Run Full System Analysis", use_container_width=True):
        with st.spinner("HUNTER Sr is analysing the entire system... This may take a moment."):
            try:
                report, suggestions, metrics = run_overseer()
                st.success("Analysis complete!")
                st.rerun()
            except Exception as e:
                st.error(f"Analysis failed: {e}")


# ===========================================================================
# FLOATING CHAT — HUNTER Jr
# ===========================================================================

# ===========================================================================
# HUNTER Jr — Sidebar Chat
# ===========================================================================
if "jr_history" not in st.session_state:
    st.session_state.jr_history = []
if "jr_open" not in st.session_state:
    st.session_state.jr_open = False



# ===========================================================================
# SECTION 6: HUNTER Jr (Chat)
# ===========================================================================
with section_tab6:
    st.markdown("")
    jr_col1, jr_col2, jr_col3 = st.columns([1, 2, 1])
    with jr_col2:
        st.markdown("""
        <div style="text-align:center;padding:30px 0 10px 0;">
            <div style="width:60px;height:60px;background:linear-gradient(135deg,#7C3AED,#5B21B6);border-radius:50%;display:inline-flex;align-items:center;justify-content:center;font-size:1.8em;">🐣</div>
            <div style="font-weight:700;color:#7C3AED !important;font-size:1.4em;margin-top:10px;">HUNTER Jr</div>
            <div style="color:#6B7280 !important;font-size:0.9em;margin-top:4px;">Your AI research assistant</div>
            <div style="border-top:2px solid #7C3AED;width:60px;margin:16px auto 0 auto;"></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # Messages
    jr_msg_col1, jr_msg_col2, jr_msg_col3 = st.columns([1, 3, 1])
    with jr_msg_col2:
        for msg in st.session_state.jr_history[-10:]:
            if msg["role"] == "user":
                st.markdown(f"""<div style="background:#F5F3FF;padding:12px 16px;border-radius:16px 16px 4px 16px;margin:8px 0 8px 60px;border:1px solid #E9D5FF;">
                    <span style="color:#374151 !important;font-size:0.92em;">{msg['content']}</span></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div style="background:#FAFBFC;padding:12px 16px;border-radius:16px 16px 16px 4px;margin:8px 60px 8px 0;border:1px solid #E5E7EB;border-left:3px solid #7C3AED;">
                    <strong style="color:#7C3AED !important;font-size:0.85em;">🐣 HUNTER Jr</strong><br>
                    <span style="color:#374151 !important;font-size:0.92em;">{msg['content']}</span></div>""", unsafe_allow_html=True)

        if not st.session_state.jr_history:
            st.markdown("""
            <div style="text-align:center;padding:60px 20px;color:#9CA3AF !important;">
                <div style="font-size:1.1em;margin-bottom:12px;">Ask me anything about HUNTER</div>
                <div style="font-size:0.85em;line-height:1.8;">
                    "What's my best diamond?"<br>
                    "How is the portfolio doing?"<br>
                    "Which domain produces the most collisions?"<br>
                    "Any positions about to expire?"
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        # Use a form to prevent the rerun loop — form only submits on Enter/button click
        with st.form("jr_chat_form", clear_on_submit=True):
            jr_input = st.text_input("Message HUNTER Jr...", placeholder="Ask anything about your data, portfolio, or system...", label_visibility="collapsed")
            jr_submitted = st.form_submit_button("Send", use_container_width=True)

        if jr_submitted and jr_input:
            st.session_state.jr_history.append({"role": "user", "content": jr_input})
            with st.spinner("🐣 Thinking..."):
                try:
                    jr_response = chat_with_hunter(jr_input, st.session_state.jr_history)
                    st.session_state.jr_history.append({"role": "assistant", "content": jr_response})
                    st.rerun()
                except Exception as e:
                    st.session_state.jr_history.append({"role": "assistant", "content": f"Oops: {e}"})
                    st.rerun()

        if st.session_state.jr_history:
            if st.button("Clear Chat", key="jr_clear"):
                st.session_state.jr_history = []
                st.rerun()

        st.markdown("")
        st.caption("AI can make mistakes. Verify important data.")


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------
st.components.v1.html('<script>setTimeout(function(){window.location.reload();},60000);</script>', height=0)
