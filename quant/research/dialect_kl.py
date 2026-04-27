"""Dialect KL-divergence estimator — measure how DIFFERENT two specialist languages are.

Theory
======

The Universal Translator Theorem (conjectured) says:

    alpha_compositional <= C * D_KL(dialect_A || dialect_B)

where D_KL is the Kullback-Leibler divergence between the marginal
distributions of statements in each dialect. If two dialects describe the
same reality very differently (high KL), there's more compositional alpha
to extract by translating between them.

Practical estimation
====================

We don't have direct access to the distributions over statements, only to
samples of statements (the facts in each silo of the corpus). We approximate
via embedding distributions:

  1. Take a representative sample of facts from each silo.
  2. Embed each via sentence-transformers (all-MiniLM-L6-v2, 384-dim).
  3. Treat each silo's embeddings as samples from a multivariate Gaussian.
  4. Compute closed-form KL between the two Gaussians:

         D_KL(N_a || N_b) = 0.5 * [
             tr(Sigma_b^{-1} Sigma_a)
             + (mu_b - mu_a)^T Sigma_b^{-1} (mu_b - mu_a)
             - k
             + log(det Sigma_b / det Sigma_a)
         ]

This is an approximation — the embedding distributions are not actually
Gaussian — but it's a defensible first-order proxy and it's what a real
information-theoretic translation theorem would compute.

Output
======

For each pair of silos (silo_A, silo_B):

  D_KL(silo_A || silo_B)   # asymmetric: how surprising is silo A given silo B's distribution?
  D_KL(silo_B || silo_A)
  D_JS(silo_A, silo_B)     # symmetric Jensen-Shannon for clustering

The full N x N matrix becomes HUNTER's data-derived dialect-distance matrix.
This replaces the hand-calibrated 153-pair distance matrix with an empirical,
data-driven, automatically-updating one.

This is the empirical instrument that tests the Universal Translator Theorem.
If diamond-tier theses concentrate in high-KL silo pairs, the theorem is
supported. If not, the theorem needs revision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


# ============================================================
# Embedding wrapper (lazy-loaded)
# ============================================================

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer

        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    """Return (n, 384) array of L2-normalised embeddings."""
    if not texts:
        return np.zeros((0, 384))
    return np.asarray(
        _get_embed_model().encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    )


# ============================================================
# Distribution fitting and KL
# ============================================================

@dataclass
class DialectDistribution:
    silo: str
    n_samples: int
    mean: np.ndarray
    cov: np.ndarray  # diagonal-regularised covariance
    raw_samples: np.ndarray  # for diagnostics / cluster analysis

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])


def fit_dialect_distribution(
    silo: str,
    statements: Sequence[str],
    regularisation: float = 1e-3,
) -> DialectDistribution:
    """Embed silo statements; fit Gaussian with diagonal regularisation.

    The diagonal regularisation prevents singular covariance when fewer
    samples than dimensions (which is typical: 384-dim embeddings, ~100
    samples per silo).
    """
    if len(statements) < 4:
        raise ValueError(f"need >= 4 statements to fit a distribution; got {len(statements)}")
    X = embed_texts(statements)
    mu = X.mean(axis=0)
    cov = np.cov(X.T)
    # Regularise: shrink toward diagonal
    diag = np.diag(np.diag(cov))
    cov_reg = (1 - regularisation) * cov + regularisation * (
        np.eye(X.shape[1]) * np.trace(diag) / X.shape[1]
    )
    return DialectDistribution(
        silo=silo,
        n_samples=len(statements),
        mean=mu,
        cov=cov_reg,
        raw_samples=X,
    )


def kl_gaussian(
    mu_a: np.ndarray, cov_a: np.ndarray,
    mu_b: np.ndarray, cov_b: np.ndarray,
) -> float:
    """Closed-form KL(N_a || N_b) for multivariate Gaussians.

    Returns >= 0 in nats. Use to compare dialects via their embedding
    distributions.
    """
    k = mu_a.shape[0]
    diff = mu_b - mu_a
    # Use pseudo-inverse to handle near-singular cov_b
    inv_cov_b = np.linalg.pinv(cov_b)
    trace_term = float(np.trace(inv_cov_b @ cov_a))
    quad_term = float(diff @ inv_cov_b @ diff)
    sign_a, log_det_a = np.linalg.slogdet(cov_a)
    sign_b, log_det_b = np.linalg.slogdet(cov_b)
    # Guard against numerical disasters from singular covariance
    if sign_a <= 0 or sign_b <= 0:
        log_det_term = 0.0
    else:
        log_det_term = float(log_det_b - log_det_a)
    return 0.5 * (trace_term + quad_term - k + log_det_term)


def jensen_shannon(
    dist_a: DialectDistribution,
    dist_b: DialectDistribution,
) -> float:
    """Symmetric JS-divergence between two dialect distributions, in nats.

    JS(A, B) = 0.5 * D_KL(A || M) + 0.5 * D_KL(B || M)  with M = 0.5 * (A + B)

    Approximated under the Gaussian fit by averaging the means and covariances.
    """
    mu_m = 0.5 * (dist_a.mean + dist_b.mean)
    cov_m = 0.5 * (dist_a.cov + dist_b.cov)
    kl_a_m = kl_gaussian(dist_a.mean, dist_a.cov, mu_m, cov_m)
    kl_b_m = kl_gaussian(dist_b.mean, dist_b.cov, mu_m, cov_m)
    return 0.5 * (kl_a_m + kl_b_m)


# ============================================================
# Pairwise matrix
# ============================================================

@dataclass
class DialectKLMatrix:
    silos: list[str]
    asymmetric_kl: np.ndarray  # M[i, j] = D_KL(silo_i || silo_j)
    jensen_shannon: np.ndarray
    n_per_silo: dict[str, int]

    def to_dict(self) -> dict:
        return {
            "silos": self.silos,
            "asymmetric_kl": [[round(x, 4) for x in row] for row in self.asymmetric_kl],
            "jensen_shannon": [[round(x, 4) for x in row] for row in self.jensen_shannon],
            "n_per_silo": self.n_per_silo,
        }

    def top_pairs(self, k: int = 10) -> list[tuple[str, str, float]]:
        """Top-k highest-JS-divergence silo pairs (most different dialects)."""
        out: list[tuple[str, str, float]] = []
        for i in range(len(self.silos)):
            for j in range(i + 1, len(self.silos)):
                out.append(
                    (self.silos[i], self.silos[j], float(self.jensen_shannon[i, j]))
                )
        out.sort(key=lambda x: -x[2])
        return out[:k]


def compute_silo_kl_matrix(
    statements_by_silo: dict[str, Sequence[str]],
    regularisation: float = 1e-3,
) -> DialectKLMatrix:
    """For each silo, fit a dialect distribution; compute pairwise KL + JS.

    Inputs:
      statements_by_silo: dict mapping silo name -> list of representative
        statements from that silo (typically ~50-200 per silo)

    Returns DialectKLMatrix.
    """
    silos = sorted(statements_by_silo)
    distributions: dict[str, DialectDistribution] = {}
    n_per_silo: dict[str, int] = {}
    for silo in silos:
        try:
            distributions[silo] = fit_dialect_distribution(
                silo, statements_by_silo[silo], regularisation=regularisation,
            )
            n_per_silo[silo] = len(statements_by_silo[silo])
        except ValueError:
            n_per_silo[silo] = len(statements_by_silo[silo])

    silos_with_dist = [s for s in silos if s in distributions]
    n = len(silos_with_dist)
    asym = np.zeros((n, n))
    js = np.zeros((n, n))

    for i, sa in enumerate(silos_with_dist):
        for j, sb in enumerate(silos_with_dist):
            if i == j:
                asym[i, j] = 0.0
                js[i, j] = 0.0
                continue
            asym[i, j] = kl_gaussian(
                distributions[sa].mean, distributions[sa].cov,
                distributions[sb].mean, distributions[sb].cov,
            )
            if j > i:
                js_val = jensen_shannon(distributions[sa], distributions[sb])
                js[i, j] = js_val
                js[j, i] = js_val

    return DialectKLMatrix(
        silos=silos_with_dist,
        asymmetric_kl=asym,
        jensen_shannon=js,
        n_per_silo=n_per_silo,
    )
