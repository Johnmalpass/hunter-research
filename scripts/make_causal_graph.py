"""
Regenerate docs/img/causal_graph.png from the frozen corpus.

203 methodology nodes, 171 directed edges. The visualisation emphasises
what's structurally important: the single degree-9 hub (ARGUS Enterprise
DCF cap-rate assumptions), the few degree-3 secondary hubs, and the
depleted middle (degrees 4-8 are empty) that distinguishes hub-and-spoke
from a scale-free power law.

Reads from the Zenodo SQLite. Writes to docs/img/causal_graph.png.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx


CORPUS = Path("/tmp/hunter_zenodo/hunter_corpus_v1/hunter_corpus_v1.sqlite")
OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "causal_graph.png"


def _short(name: str, limit: int = 46) -> str:
    if "|" in name:
        name = name.split("|", 1)[0]
    name = name.strip()
    if len(name) <= limit:
        return name
    return name[: limit - 1].rstrip() + "…"


def build_graph() -> nx.DiGraph:
    conn = sqlite3.connect(CORPUS)
    rows = conn.execute(
        "SELECT cause_node, effect_node FROM causal_edges"
    ).fetchall()
    conn.close()
    g = nx.DiGraph()
    for cause, effect in rows:
        g.add_edge(cause.strip(), effect.strip())
    return g


def render(g: nx.DiGraph) -> None:
    degree = dict(g.degree())
    max_deg = max(degree.values())
    hub = max(degree, key=degree.get)

    pos = nx.spring_layout(g, k=0.85, iterations=220, seed=13)

    # node styling by role
    sizes, colors, edgecolors = [], [], []
    for n in g.nodes():
        d = degree[n]
        if n == hub:
            sizes.append(1600)
            colors.append("#C89B3C")
            edgecolors.append("#1F2A35")
        elif d >= 3:
            sizes.append(340)
            colors.append("#4A6E8A")
            edgecolors.append("#1F2A35")
        elif d == 2:
            sizes.append(44)
            colors.append("#7F8A96")
            edgecolors.append("none")
        else:
            sizes.append(14)
            colors.append("#B6BCC3")
            edgecolors.append("none")

    hub_edges = [(u, v) for (u, v) in g.edges() if u == hub or v == hub]
    other_edges = [e for e in g.edges() if u_v_not_hub(e, hub)]

    fig, ax = plt.subplots(figsize=(12.2, 8.2), dpi=160)
    ax.set_facecolor("#FAFAF7")
    fig.patch.set_facecolor("#FAFAF7")

    # faint peripheral edges
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=other_edges,
        ax=ax,
        edge_color="#CFD3D8",
        width=0.45,
        alpha=0.55,
        arrows=False,
    )
    # hub spokes emphasised
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=hub_edges,
        ax=ax,
        edge_color="#8C6A28",
        width=1.6,
        alpha=0.9,
        arrows=False,
    )

    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=colors,
        edgecolors=edgecolors,
        linewidths=0.9,
    )

    # secondary-hub labels
    sec_labels = {n: _short(n, 42) for n, d in degree.items() if d >= 3 and n != hub}
    nx.draw_networkx_labels(
        g,
        pos,
        labels=sec_labels,
        ax=ax,
        font_size=8.4,
        font_family="DejaVu Sans",
        font_color="#1F2A35",
        bbox=dict(
            facecolor="#FAFAF7",
            edgecolor="#C8CDD3",
            boxstyle="round,pad=0.30",
            alpha=0.95,
        ),
    )

    # hub label: larger gold-bordered box above the hub node
    hx, hy = pos[hub]
    ax.text(
        hx,
        hy + 0.10,
        _short(hub, 48),
        ha="center",
        va="bottom",
        fontsize=10.8,
        fontweight="semibold",
        color="#1F2A35",
        bbox=dict(
            facecolor="#FFF6DF",
            edgecolor="#C89B3C",
            boxstyle="round,pad=0.42",
            linewidth=1.2,
        ),
        zorder=10,
    )

    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()
    ax.set_title(
        "Methodology causal graph · hub-and-spoke around ARGUS Enterprise DCF",
        fontsize=13.8,
        fontweight="semibold",
        color="#1F2A35",
        pad=14,
    )
    ax.text(
        0.5,
        -0.035,
        f"{n_nodes} nodes · {n_edges} directed edges · "
        f"depleted middle at degrees 4–8 · single degree-{max_deg} hub",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
        color="#4A5360",
    )

    ax.set_axis_off()
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"wrote {OUT}  ({n_nodes} nodes, {n_edges} edges, hub degree {max_deg})")


def u_v_not_hub(edge, hub) -> bool:
    return edge[0] != hub and edge[1] != hub


if __name__ == "__main__":
    render(build_graph())
