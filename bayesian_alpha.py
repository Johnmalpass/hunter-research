"""Bayesian re-analysis of the pre-freeze HUNTER corpus.

Runs the inference framework specified in `docs/STATISTICAL_METHODS.md` against
the frozen Zenodo corpus. Reports posterior probabilities, credible intervals,
and Bayes factors for the two pre-freeze patterns that have enough data:

  1. Narrative-survival correlation (r negative)
  2. Cross-silo > within-silo score difference (Layer 1 prediction)

This is a supplement to the frequentist tests in `MATH_VERIFICATION.md`.
Same data, different inferential framework.

Run:
    python bayesian_alpha.py
    python bayesian_alpha.py --db /path/to/hunter_corpus_v1.sqlite
    python bayesian_alpha.py --n-samples 50000

No API calls. Pure NumPy + SciPy. Runs in under 10 seconds.
"""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from pathlib import Path
from statistics import mean

import numpy as np
from scipy import stats

# Default to the Zenodo extract location; allow override
DEFAULT_DB = "/tmp/hunter_zenodo/hunter_corpus_v1/hunter_corpus_v1.sqlite"


# ============================================================
# Narrative score regex (mirrors narrative_detector.py exactly)
# ============================================================

import re

PROTAGONIST_PATTERNS = [
    r"\b[A-Z][A-Z0-9&]{1,5}\b",
    r"\b(?:[A-Z][a-z]+\s+)+(?:Inc|Corp|Corporation|Ltd|LLC|REIT|Group|Holdings|Industries|Company)\b",
    r"\b(?:the\s+)?(?:pension|insurance|bank|hotel|office|industrial)\s+\w+\s+(?:sector|industry|market|companies|REITs?)\b",
]
ANTAGONIST_PATTERNS = [
    r"\b(?:regulator|regulation|rule|requirement|mandate|framework|model|formula|methodology|assumption)\b",
    r"\b(?:FERC|NAIC|SEC|FDA|EU|CMS|FASB|IASB|OSHA|EPA|CFTC|Federal Reserve)\b",
    r"\b(?:model|assumption)s?\s+(?:is|are|remain)\s+(?:wrong|outdated|stale|broken|invalid)\b",
]
COMPLICATION_PATTERNS = [
    r"\bmispric(?:e|ing|ed)\b",
    r"\bunderestimat(?:e|ing|ed)\b",
    r"\boverestimat(?:e|ing|ed)\b",
    r"\barbitrage\b",
    r"\b(?:systematically|materially|significantly)\s+(?:wrong|incorrect|miscalibrated)\b",
    r"\b(?:gap|mismatch|discrepancy|inconsistency|divergence)\b",
]
CATALYST_PATTERNS = [
    r"\b(?:20\d{2})\b",
    r"\bQ[1-4]\s*20\d{2}\b",
    r"\b(?:March|April|May|June|July|August|September|October|November|December|January|February)\s+20\d{2}\b",
    r"\b(?:by|before|after|on|effective|within)\s+(?:\d+\s+(?:days?|weeks?|months?))\b",
    r"\b(?:maturity wall|catalyst|trigger|deadline|effective date)\b",
]
RESOLUTION_PATTERNS = [
    r"\b(?:repric(?:e|ing|ed)|correct(?:ion|ed|ing)?|revalu(?:e|ation|ed)|downgrad(?:e|ed)|writedown|impairment)\b",
    r"\b(?:settlement|adjustment|revision|amendment|rebase)\b",
    r"\b(?:converge|narrow|close|widen|compress|expand)\s+(?:spread|gap|differential)\b",
]


def _density(text: str, patterns: list, cap: int = 3) -> float:
    found = set()
    for rx in patterns:
        for m in re.findall(rx, text, flags=re.IGNORECASE):
            found.add(m.lower() if isinstance(m, str) else str(m))
    return min(1.0, len(found) / cap)


def narrative_strength(text: str) -> float:
    text = (text or "")[:5000]
    parts = [
        _density(text, PROTAGONIST_PATTERNS),
        _density(text, ANTAGONIST_PATTERNS),
        _density(text, COMPLICATION_PATTERNS),
        _density(text, CATALYST_PATTERNS),
        _density(text, RESOLUTION_PATTERNS),
    ]
    return mean(parts)


# ============================================================
# Data loaders (Zenodo corpus)
# ============================================================


def load_combined_hypotheses(db_path: str) -> list[dict]:
    """Pull all hypotheses (main + archive) with text + survival + score."""
    conn = sqlite3.connect(db_path)
    rows = []
    for table in ("hypotheses", "hypotheses_archive"):
        try:
            for hid, text, surv, score in conn.execute(
                f"SELECT id, hypothesis_text, survived_kill, diamond_score "
                f"FROM {table} WHERE hypothesis_text IS NOT NULL"
            ):
                rows.append({
                    "id": hid,
                    "table": table,
                    "text": text,
                    "survived": int(surv or 0),
                    "score": float(score) if score is not None else None,
                })
        except sqlite3.OperationalError as e:
            print(f"warning: could not read {table}: {e}", file=sys.stderr)
    conn.close()
    return rows


def load_silo_counts(db_path: str) -> dict[int, int]:
    """Map collision_id -> num_domains. Used to bucket scored hypotheses by silo count."""
    conn = sqlite3.connect(db_path)
    out = {}
    for cid, n in conn.execute("SELECT id, num_domains FROM collisions WHERE num_domains IS NOT NULL"):
        out[cid] = n
    conn.close()
    return out


def load_scored_with_silos(db_path: str) -> list[tuple[int, float]]:
    """Return (num_silos, diamond_score) pairs for every scored hypothesis with a valid collision link."""
    conn = sqlite3.connect(db_path)
    out = []
    for table in ("hypotheses", "hypotheses_archive"):
        try:
            for n, s in conn.execute(
                f"SELECT col.num_domains, h.diamond_score "
                f"FROM {table} h JOIN collisions col ON col.id = h.collision_id "
                f"WHERE h.diamond_score IS NOT NULL AND col.num_domains IS NOT NULL"
            ):
                out.append((int(n), float(s)))
        except sqlite3.OperationalError:
            pass
    conn.close()
    return out


# ============================================================
# Inference 1: Posterior on narrative-survival correlation
# ============================================================


def posterior_correlation_via_fisher_z(
    x: list[float], y: list[float], n_samples: int = 10000, seed: int = 42
) -> dict:
    """Bayesian posterior on Pearson r via Fisher z-transform.

    Prior on r: uniform on [-1, 1] (translates to a Cauchy-ish prior on z).
    Likelihood: Fisher's z is approximately Normal with variance 1/(n-3).

    Returns posterior mean, 95% credible interval, P(r < 0), and Bayes factor.
    """
    n = len(x)
    if n < 4:
        return {"error": f"n={n} too small for Fisher z"}

    r_obs = float(np.corrcoef(x, y)[0, 1])
    z_obs = 0.5 * np.log((1 + r_obs) / (1 - r_obs))
    se_z = 1.0 / math.sqrt(n - 3)

    rng = np.random.default_rng(seed)
    z_post = rng.normal(z_obs, se_z, size=n_samples)
    r_post = np.tanh(z_post)

    ci_low, ci_high = np.percentile(r_post, [2.5, 97.5])
    p_negative = float(np.mean(r_post < 0))
    p_below_minus_0_2 = float(np.mean(r_post < -0.2))

    # Savage-Dickey Bayes factor for H0: r=0 vs H1: r != 0
    # Under uniform prior on r in [-1,1], prior density at r=0 is 0.5
    # Posterior density at r=0 estimated via kernel density
    from scipy.stats import gaussian_kde
    posterior_density_at_zero = float(gaussian_kde(r_post)(0))
    bf_10 = 0.5 / posterior_density_at_zero  # BF in favour of H1

    return {
        "n": n,
        "r_observed": r_obs,
        "posterior_mean_r": float(np.mean(r_post)),
        "ci_95": (float(ci_low), float(ci_high)),
        "P(r < 0)": p_negative,
        "P(r < -0.2)": p_below_minus_0_2,
        "BF_10_savage_dickey": bf_10,
    }


# ============================================================
# Inference 2: Posterior on cross-silo vs within-silo score difference
# ============================================================


def posterior_group_difference_normal_normal(
    group_a: list[float], group_b: list[float],
    prior_mean: float = 0.0, prior_sd: float = 20.0,
    n_samples: int = 10000, seed: int = 42,
) -> dict:
    """Bayesian posterior on the difference in means between two independent groups.

    Each group's likelihood is Normal(mu, sigma^2) with sigma estimated from the
    pooled sample. Prior on each mu is Normal(prior_mean, prior_sd^2).

    Returns posterior on (mu_b - mu_a) including 95% CI and P(mu_b > mu_a).
    """
    n_a, n_b = len(group_a), len(group_b)
    if n_a < 2 or n_b < 2:
        return {"error": f"n_a={n_a}, n_b={n_b} too small"}

    a = np.array(group_a)
    b = np.array(group_b)

    # Pooled sigma estimate
    sigma_pooled = math.sqrt(
        ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / (n_a + n_b - 2)
    )

    # Posterior on each group mean is Normal-Normal conjugate
    # Prior: mu ~ N(prior_mean, prior_sd^2), Likelihood: x_i ~ N(mu, sigma_pooled^2)
    # Posterior: mu | data ~ N(post_mean, post_var)
    def posterior(group_x, n):
        prior_var = prior_sd ** 2
        like_var = (sigma_pooled ** 2) / n
        post_var = 1.0 / (1.0 / prior_var + 1.0 / like_var)
        post_mean = post_var * (prior_mean / prior_var + group_x.mean() / like_var)
        return post_mean, math.sqrt(post_var)

    pa_mean, pa_sd = posterior(a, n_a)
    pb_mean, pb_sd = posterior(b, n_b)

    rng = np.random.default_rng(seed)
    diff_samples = (
        rng.normal(pb_mean, pb_sd, size=n_samples)
        - rng.normal(pa_mean, pa_sd, size=n_samples)
    )

    ci_low, ci_high = np.percentile(diff_samples, [2.5, 97.5])
    p_b_greater = float(np.mean(diff_samples > 0))
    p_diff_above_5 = float(np.mean(diff_samples > 5))

    return {
        "n_a": n_a, "n_b": n_b,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "observed_diff": float(b.mean() - a.mean()),
        "posterior_mean_diff": float(np.mean(diff_samples)),
        "ci_95": (float(ci_low), float(ci_high)),
        "P(mu_b > mu_a)": p_b_greater,
        "P(diff > 5 points)": p_diff_above_5,
        "sigma_pooled": sigma_pooled,
    }


# ============================================================
# Reporting
# ============================================================


def fmt_ci(ci):
    return f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"


def report_correlation(result: dict, label: str):
    print(f"\n{label}")
    print("-" * 60)
    if "error" in result:
        print(f"  skipped: {result['error']}")
        return
    print(f"  n                       {result['n']}")
    print(f"  observed r              {result['r_observed']:+.4f}")
    print(f"  posterior mean r        {result['posterior_mean_r']:+.4f}")
    print(f"  95% credible interval   {fmt_ci(result['ci_95'])}")
    print(f"  P(r < 0)                {result['P(r < 0)']:.4f}")
    print(f"  P(r < -0.2)             {result['P(r < -0.2)']:.4f}")
    print(f"  Bayes factor BF_10      {result['BF_10_savage_dickey']:.2f}")
    interp = (
        "extreme evidence against null" if result['BF_10_savage_dickey'] >= 100
        else "very strong evidence" if result['BF_10_savage_dickey'] >= 30
        else "strong evidence" if result['BF_10_savage_dickey'] >= 10
        else "moderate evidence" if result['BF_10_savage_dickey'] >= 3
        else "weak evidence" if result['BF_10_savage_dickey'] >= 1
        else "evidence for null"
    )
    print(f"  interpretation          {interp} (Kass-Raftery scale)")


def report_difference(result: dict, label: str):
    print(f"\n{label}")
    print("-" * 60)
    if "error" in result:
        print(f"  skipped: {result['error']}")
        return
    print(f"  n_a={result['n_a']}, n_b={result['n_b']}")
    print(f"  observed means          a={result['mean_a']:.2f}, b={result['mean_b']:.2f}")
    print(f"  observed difference     {result['observed_diff']:+.2f} points")
    print(f"  posterior mean diff     {result['posterior_mean_diff']:+.3f}")
    print(f"  95% credible interval   {fmt_ci(result['ci_95'])}")
    print(f"  P(b > a)                {result['P(mu_b > mu_a)']:.4f}")
    print(f"  P(diff > 5 points)      {result['P(diff > 5 points)']:.4f}")
    print(f"  sigma (pooled)          {result['sigma_pooled']:.2f}")


# ============================================================
# Main
# ============================================================


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Bayesian re-analysis of HUNTER pre-freeze corpus.",
    )
    parser.add_argument("--db", default=DEFAULT_DB, help=f"path to corpus sqlite (default {DEFAULT_DB})")
    parser.add_argument("--n-samples", type=int, default=10000, help="posterior samples (default 10000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42)")
    args = parser.parse_args(argv)

    if not Path(args.db).exists():
        print(f"error: corpus not found at {args.db}", file=sys.stderr)
        print("hint: extract the Zenodo release to /tmp/hunter_zenodo/ first", file=sys.stderr)
        return 2

    print("=" * 60)
    print("HUNTER Bayesian re-analysis of pre-freeze corpus")
    print("=" * 60)
    print(f"corpus:     {args.db}")
    print(f"samples:    {args.n_samples}")
    print(f"seed:       {args.seed}")

    # Load data
    rows = load_combined_hypotheses(args.db)
    print(f"loaded:     {len(rows)} hypotheses across both pipelines")

    # ============================================================
    # Test 1: narrative -> survival correlation
    # ============================================================
    ns = [narrative_strength(r["text"]) for r in rows]
    surv = [r["survived"] for r in rows]
    result_corr = posterior_correlation_via_fisher_z(
        ns, surv, n_samples=args.n_samples, seed=args.seed
    )
    report_correlation(
        result_corr,
        "Test 1: narrative strength -> kill survival (combined 324)",
    )

    # ============================================================
    # Test 2: cross-silo > within-silo score difference
    # ============================================================
    silo_score_pairs = load_scored_with_silos(args.db)
    within = [s for n, s in silo_score_pairs if n == 1]
    cross = [s for n, s in silo_score_pairs if n >= 2]
    result_diff = posterior_group_difference_normal_normal(
        within, cross, n_samples=args.n_samples, seed=args.seed,
    )
    report_difference(
        result_diff,
        "Test 2: cross-silo score (b) vs within-silo score (a), Layer 1",
    )

    # ============================================================
    # Test 3: hump-curve cross-check, d=2 vs d>=4
    # ============================================================
    d2 = [s for n, s in silo_score_pairs if n == 2]
    d4plus = [s for n, s in silo_score_pairs if n >= 4]
    result_hump = posterior_group_difference_normal_normal(
        d4plus, d2, n_samples=args.n_samples, seed=args.seed,
    )
    report_difference(
        result_hump,
        "Test 3: depth 2 score (b) vs depth >= 4 score (a), Layer 7 hump check",
    )

    print("\n" + "=" * 60)
    print("notes")
    print("=" * 60)
    print("- Tests 1 and 2 use the combined 324 corpus.")
    print("- Test 3 cross-checks the hump-curve prediction (d=2 should beat d>=4).")
    print("- All priors are weakly informative; see docs/STATISTICAL_METHODS.md.")
    print("- These are descriptive Bayesian summaries on the pre-freeze data.")
    print("- The summer 2026 study is the primary out-of-sample test.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
