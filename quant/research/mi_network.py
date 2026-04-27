"""Mechanism Mutual Information Network.

For all mechanisms with closed signals in the ledger, build a graph where
edges are the mutual information between mechanisms' daily signal time
series. Run greedy modularity community detection. Output:

  - per-pair mutual information in bits
  - clusters of mechanisms that say the same thing
  - a portfolio diversity score derived from inter-cluster vs intra-cluster MI

Why this matters
================

A portfolio of 18 mechanisms is *not* 18-mechanism diversification if 12 of
them touch CMBS+insurance and fire on the same days. The MI network reveals
this redundancy automatically. Two mechanisms in the same cluster should
not double-vote in the coalition; mechanisms across clusters provide
independent evidence and deserve full weight.

This is ensemble theory + information theory + network analysis applied to
compiled financial theses. As far as we have searched, this combination
is not in published quant literature.

The graph and clustering update online: each time a mechanism closes a
new signal, the time series grows, the MI re-estimates, the clustering
re-emits. The TRADER agent reads the latest clustering at every cycle.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from quant.data.base import DEFAULT_DB
from quant.research.synergy import discrete_mi


@dataclass
class MIEdge:
    src: str
    dst: str
    mi_bits: float
    overlap_days: int


@dataclass
class MINetworkResult:
    asof: datetime
    mechanisms: list[str]
    edges: list[MIEdge]
    clusters: dict[str, str]  # mechanism_id -> cluster_label
    diversity_score: float
    n_clusters: int


def _signal_dates_by_mechanism(
    db_path: Path | str | None,
    lookback_days: int,
) -> dict[str, set]:
    """Return {mechanism_id: set(date strings)} for signals in the lookback."""
    path = Path(db_path) if db_path else DEFAULT_DB
    if not path.exists():
        return {}
    cutoff = (
        datetime.now(timezone.utc) - timedelta(days=lookback_days)
    ).isoformat()
    conn = sqlite3.connect(str(path))
    try:
        try:
            rows = conn.execute(
                "SELECT mechanism_id, asof FROM mechanism_signals "
                "WHERE asof >= ?",
                (cutoff,),
            ).fetchall()
        except sqlite3.OperationalError:
            return {}
    finally:
        conn.close()
    out: dict[str, set] = {}
    for mid, asof in rows:
        out.setdefault(mid, set()).add(asof[:10])  # YYYY-MM-DD
    return out


def _binary_series(date_set: set, all_dates: list[str]) -> np.ndarray:
    """Daily binary series: 1 if mechanism fired that day, 0 otherwise."""
    return np.array([1 if d in date_set else 0 for d in all_dates], dtype=int)


def compute_mi_network(
    db_path: Path | str | None = None,
    *,
    lookback_days: int = 180,
    min_overlap_days: int = 30,
    mi_edge_threshold_bits: float = 0.05,
) -> MINetworkResult:
    """Build the mechanism MI network from the ledger.

    Returns an `MINetworkResult` with per-pair MI, cluster assignments, and
    a portfolio diversity score in [0, 1] (higher = more diverse).
    """
    sig_dates = _signal_dates_by_mechanism(db_path, lookback_days)
    mechanisms = sorted(sig_dates)

    if len(mechanisms) < 2:
        return MINetworkResult(
            asof=datetime.now(timezone.utc),
            mechanisms=mechanisms,
            edges=[],
            clusters={m: m for m in mechanisms},
            diversity_score=1.0,  # single mechanism is "fully diverse"
            n_clusters=len(mechanisms),
        )

    # Universe: every calendar day in the lookback window. NOT just days when
    # something fired -- using firing days only collapses correlated
    # mechanisms to all-1s vectors with zero variance and no recoverable MI.
    end = datetime.now(timezone.utc)
    earliest_signal = min(min(s) for s in sig_dates.values()) if sig_dates else None
    if earliest_signal:
        start = max(
            datetime.fromisoformat(earliest_signal).replace(tzinfo=timezone.utc),
            end - timedelta(days=lookback_days),
        )
    else:
        start = end - timedelta(days=lookback_days)
    all_dates: list[str] = []
    cur = start
    while cur.date() <= end.date():
        all_dates.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    if len(all_dates) < min_overlap_days:
        return MINetworkResult(
            asof=datetime.now(timezone.utc),
            mechanisms=mechanisms,
            edges=[],
            clusters={m: m for m in mechanisms},
            diversity_score=1.0,
            n_clusters=len(mechanisms),
        )

    # Pairwise MI on binary daily series
    series = {m: _binary_series(sig_dates[m], all_dates) for m in mechanisms}
    edges: list[MIEdge] = []
    for i, ma in enumerate(mechanisms):
        for mb in mechanisms[i + 1:]:
            mi = discrete_mi(series[ma], series[mb])
            if mi >= mi_edge_threshold_bits:
                edges.append(
                    MIEdge(
                        src=ma, dst=mb, mi_bits=mi,
                        overlap_days=int((series[ma] & series[mb]).sum()),
                    )
                )

    # Community detection via greedy modularity (no scikit-learn)
    clusters = _greedy_modularity_communities(mechanisms, edges)
    n_clusters = len({clusters[m] for m in mechanisms})
    diversity = _diversity_score(mechanisms, edges, clusters)

    return MINetworkResult(
        asof=datetime.now(timezone.utc),
        mechanisms=mechanisms,
        edges=edges,
        clusters=clusters,
        diversity_score=diversity,
        n_clusters=n_clusters,
    )


def _greedy_modularity_communities(
    nodes: list[str],
    edges: list[MIEdge],
) -> dict[str, str]:
    """Use networkx's greedy_modularity_communities (no scikit-learn dep)."""
    try:
        import networkx as nx
        from networkx.algorithms.community import greedy_modularity_communities
    except ImportError:
        # Fallback: each node its own cluster
        return {n: n for n in nodes}

    G = nx.Graph()
    G.add_nodes_from(nodes)
    for e in edges:
        G.add_edge(e.src, e.dst, weight=e.mi_bits)

    if G.number_of_edges() == 0:
        return {n: n for n in nodes}

    try:
        communities = list(greedy_modularity_communities(G, weight="weight"))
    except Exception:
        return {n: n for n in nodes}

    out: dict[str, str] = {}
    for i, comm in enumerate(communities):
        label = f"cluster_{i}"
        for node in comm:
            out[node] = label
    # Singletons that fell out of communities
    for n in nodes:
        out.setdefault(n, "cluster_singleton")
    return out


def _diversity_score(
    nodes: list[str],
    edges: list[MIEdge],
    clusters: dict[str, str],
) -> float:
    """Diversity in [0, 1].

    Defined as 1 - (weighted intra-cluster MI) / (weighted total MI).
    Higher = more of the MI lives between clusters = more diverse portfolio.
    """
    if not edges:
        return 1.0
    total = sum(e.mi_bits for e in edges)
    intra = sum(
        e.mi_bits for e in edges
        if clusters.get(e.src) == clusters.get(e.dst)
    )
    if total <= 0:
        return 1.0
    return max(0.0, 1.0 - intra / total)


def report(result: MINetworkResult) -> str:
    lines = []
    lines.append(f"Mechanism MI network — {len(result.mechanisms)} mechanisms")
    lines.append(f"  diversity score:  {result.diversity_score:.3f}  "
                 f"(higher = independent voices)")
    lines.append(f"  clusters:         {result.n_clusters}")
    lines.append("")
    lines.append("Cluster assignments:")
    by_cluster: dict[str, list[str]] = {}
    for m, c in result.clusters.items():
        by_cluster.setdefault(c, []).append(m)
    for c, members in sorted(by_cluster.items()):
        lines.append(f"  {c}:  {', '.join(sorted(members))}")
    lines.append("")
    lines.append("Top edges (highest MI):")
    for e in sorted(result.edges, key=lambda x: -x.mi_bits)[:10]:
        lines.append(
            f"  {e.src} <-> {e.dst}   MI={e.mi_bits:.3f} bits   "
            f"overlap={e.overlap_days}d"
        )
    return "\n".join(lines)
