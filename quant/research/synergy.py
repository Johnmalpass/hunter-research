"""Synergistic information estimator.

Operationalises HUNTER's 'compositional alpha' as a measurable scalar in bits.

Given a target X (asset return, hypothesis score, realised alpha) and two
information sources A, B drawn from independent silos, the interaction
information

    II(X; A; B) = I(X; A, B) - I(X; A) - I(X; B)

decomposes the relationship:

    II > 0   synergistic.  A and B together reveal more about X than they
                          do separately. THIS IS COMPOSITIONAL ALPHA IN BITS.
    II = 0   independent contributions. Standard multi-factor case.
    II < 0   redundant.    A and B carry overlapping information about X.

Core claim of the framework: HUNTER's diamond theses concentrate in the
II > 0 region. Within-silo theses concentrate at II <= 0. The summer 2026
study can be re-cast as a test on the sign and magnitude of II across
strata, which is sharper than the current diamond-score proxy.

Two estimators here:

    discrete_mi   plug-in MI for discrete arrays (clean, fast)
    ksg_mi        Kraskov-Stoegbauer-Grassberger continuous MI (k-NN based)

Both report MI in bits. KSG follows Kraskov, Stoegbauer, Grassberger 2004,
Eq. 8 (variant 1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

LOG2 = float(np.log(2.0))


# ============================================================
# Discretisation
# ============================================================

def _discretise(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = np.asarray(x).ravel()
    if x.dtype.kind in "iuS":
        # already integer-like; remap to dense small range
        _, inv = np.unique(x, return_inverse=True)
        return inv
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    if edges.size <= 1:
        return np.zeros_like(x, dtype=int)
    return np.digitize(x, edges[1:-1])


# ============================================================
# Discrete MI (plug-in)
# ============================================================

def discrete_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Plug-in MI estimator in bits for two discrete (or already-binned) arrays."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.size != y.size:
        raise ValueError("x and y must have same length")
    n = x.size
    if n == 0:
        return 0.0

    # Use unique remapping so histogramming has compact axes
    _, x_inv = np.unique(x, return_inverse=True)
    _, y_inv = np.unique(y, return_inverse=True)
    n_x = int(x_inv.max()) + 1
    n_y = int(y_inv.max()) + 1

    joint = np.zeros((n_x, n_y), dtype=np.float64)
    np.add.at(joint, (x_inv, y_inv), 1.0)
    p_xy = joint / n
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = p_xy / (p_x * p_y)
        log_term = np.where(p_xy > 0, np.log2(np.where(ratio > 0, ratio, 1.0)), 0.0)
    mi = float((p_xy * log_term).sum())
    return max(0.0, mi)


def _discrete_mi_with_joint_var(x: np.ndarray, ab: np.ndarray) -> float:
    """MI between scalar x and a multivariate (encoded as one composite label)."""
    if ab.ndim == 1:
        return discrete_mi(x, ab)
    # encode (a, b) as composite label
    n_a = int(ab[:, 0].max()) + 1
    composite = ab[:, 0] * (ab[:, 1].max() + 2) + ab[:, 1]
    return discrete_mi(x, composite)


# ============================================================
# Interaction information (the headline scalar)
# ============================================================

def interaction_information(
    x: Sequence[float],
    a: Sequence[float],
    b: Sequence[float],
    n_bins: int = 8,
) -> float:
    """II(X; A; B) in bits. Positive = synergistic; negative = redundant."""
    x_d = _discretise(np.asarray(x), n_bins)
    a_d = _discretise(np.asarray(a), n_bins)
    b_d = _discretise(np.asarray(b), n_bins)

    ab = np.column_stack([a_d, b_d])
    i_xab = _discrete_mi_with_joint_var(x_d, ab)
    i_xa = discrete_mi(x_d, a_d)
    i_xb = discrete_mi(x_d, b_d)
    return i_xab - i_xa - i_xb


# ============================================================
# Continuous KSG estimator (Kraskov, Stoegbauer, Grassberger 2004)
# ============================================================

def ksg_mi(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 3,
) -> float:
    """KSG continuous MI in bits. Variant 1 from the original paper.

    x, y can be 1D (n,) or 2D (n, d). For II in continuous form, run

        ksg_mi(X, np.column_stack([A, B])) - ksg_mi(X, A) - ksg_mi(X, B)
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    x = np.atleast_2d(x).T if np.asarray(x).ndim == 1 else np.asarray(x, dtype=float)
    y = np.atleast_2d(y).T if np.asarray(y).ndim == 1 else np.asarray(y, dtype=float)
    n = x.shape[0]
    if y.shape[0] != n:
        raise ValueError("x and y must have same length")
    if n <= k + 1:
        return 0.0

    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # k-th NN distance in joint space, Chebyshev (max-norm)
    eps = tree_xy.query(xy, k=k + 1, p=np.inf)[0][:, k]

    # count points strictly within eps in each marginal (excluding self)
    n_x = np.array(
        [len(tree_x.query_ball_point(x[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)]
    )
    n_y = np.array(
        [len(tree_y.query_ball_point(y[i], eps[i] - 1e-12, p=np.inf)) - 1 for i in range(n)]
    )
    n_x = np.maximum(n_x, 1)
    n_y = np.maximum(n_y, 1)

    mi_nats = float(
        digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    )
    mi_bits = mi_nats / LOG2
    return max(0.0, mi_bits)


def ksg_interaction_information(
    x: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    k: int = 3,
) -> float:
    """II(X; A; B) using continuous KSG estimators throughout."""
    a = np.atleast_2d(a).T if np.asarray(a).ndim == 1 else np.asarray(a, dtype=float)
    b = np.atleast_2d(b).T if np.asarray(b).ndim == 1 else np.asarray(b, dtype=float)
    ab = np.hstack([a, b])
    return ksg_mi(x, ab, k=k) - ksg_mi(x, a, k=k) - ksg_mi(x, b, k=k)


# ============================================================
# High-level wrapper
# ============================================================

@dataclass
class SynergyResult:
    ii_bits: float
    i_xa_bits: float
    i_xb_bits: float
    i_xab_bits: float
    n: int
    method: str

    @property
    def synergistic(self) -> bool:
        return self.ii_bits > 0

    @property
    def redundancy_bits(self) -> float:
        return -min(0.0, self.ii_bits)

    @property
    def synergy_bits(self) -> float:
        return max(0.0, self.ii_bits)


class SynergyEstimator:
    """High-level interface for measuring compositional alpha in bits.

    `measure(x, a, b)` returns a single SynergyResult.
    `measure_grouped(x, a, b, groups)` returns one SynergyResult per group
    label, which is the substrate for regime-conditional synergy: pass
    regime labels and you get one II value per regime.
    """

    MIN_GROUP_OBS = 30

    def __init__(self, method: str = "discrete", n_bins: int = 8, k: int = 3):
        if method not in ("discrete", "ksg"):
            raise ValueError("method must be 'discrete' or 'ksg'")
        self.method = method
        self.n_bins = n_bins
        self.k = k

    def measure_grouped(
        self,
        x: Sequence[float],
        a: Sequence[float],
        b: Sequence[float],
        groups: Sequence,
    ) -> dict:
        """Per-group synergy. Returns {group_label: SynergyResult}.

        Groups with fewer than `MIN_GROUP_OBS` observations are skipped and
        appear in the result dict as None values so callers can see they
        were considered but not estimated.
        """
        x_arr = np.asarray(x)
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        g_arr = np.asarray(groups)
        if not (x_arr.shape[0] == a_arr.shape[0] == b_arr.shape[0] == g_arr.shape[0]):
            raise ValueError("x, a, b, groups must have same length")

        out: dict = {}
        unique_labels = np.unique(g_arr)
        for label in unique_labels:
            mask = g_arr == label
            n = int(mask.sum())
            if n < self.MIN_GROUP_OBS:
                out[str(label)] = None
                continue
            out[str(label)] = self.measure(
                x_arr[mask], a_arr[mask], b_arr[mask]
            )
        return out

    def measure(
        self,
        x: Sequence[float],
        a: Sequence[float],
        b: Sequence[float],
    ) -> SynergyResult:
        # Preserve native dtype for discrete (integer arrays should not be
        # quantile-binned). KSG explicitly casts to float internally.
        x_arr = np.asarray(x)
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        if not (x_arr.shape[0] == a_arr.shape[0] == b_arr.shape[0]):
            raise ValueError("x, a, b must have same length")

        if self.method == "discrete":
            x_d = _discretise(x_arr, self.n_bins)
            a_d = _discretise(a_arr, self.n_bins)
            b_d = _discretise(b_arr, self.n_bins)
            ab = np.column_stack([a_d, b_d])
            i_xab = _discrete_mi_with_joint_var(x_d, ab)
            i_xa = discrete_mi(x_d, a_d)
            i_xb = discrete_mi(x_d, b_d)
        else:
            xf = x_arr.astype(float)
            af = a_arr.astype(float)
            bf = b_arr.astype(float)
            i_xab = ksg_mi(xf, np.column_stack([af, bf]), k=self.k)
            i_xa = ksg_mi(xf, af, k=self.k)
            i_xb = ksg_mi(xf, bf, k=self.k)

        ii = i_xab - i_xa - i_xb
        return SynergyResult(
            ii_bits=ii,
            i_xa_bits=i_xa,
            i_xb_bits=i_xb,
            i_xab_bits=i_xab,
            n=x_arr.shape[0],
            method=self.method,
        )


# ============================================================
# Self-demo: XOR (synergy) and mirror (redundancy)
# ============================================================

def _self_demo() -> None:
    rng = np.random.default_rng(0)
    n = 5000

    # XOR: A, B independent binary; X = A XOR B. Synergy = 1 bit.
    a = rng.integers(0, 2, size=n)
    b = rng.integers(0, 2, size=n)
    x_xor = (a ^ b).astype(int)

    est = SynergyEstimator(method="discrete")
    r_xor = est.measure(x_xor, a, b)
    print("XOR (expected: II ~ +1.0 bit, synergistic)")
    print(
        f"  I(X;A) = {r_xor.i_xa_bits:.3f}  "
        f"I(X;B) = {r_xor.i_xb_bits:.3f}  "
        f"I(X;A,B) = {r_xor.i_xab_bits:.3f}  "
        f"II = {r_xor.ii_bits:+.3f}"
    )

    # Mirror: B = A; X = A. Redundancy = -1 bit.
    a2 = rng.integers(0, 2, size=n)
    b2 = a2.copy()
    x_mir = a2.copy()

    r_mir = est.measure(x_mir, a2, b2)
    print("Mirror (expected: II ~ -1.0 bit, redundant)")
    print(
        f"  I(X;A) = {r_mir.i_xa_bits:.3f}  "
        f"I(X;B) = {r_mir.i_xb_bits:.3f}  "
        f"I(X;A,B) = {r_mir.i_xab_bits:.3f}  "
        f"II = {r_mir.ii_bits:+.3f}"
    )

    # Independent (X depends only on A; B is noise). II ~ 0.
    a3 = rng.integers(0, 2, size=n)
    b3 = rng.integers(0, 2, size=n)
    x_ind = a3.copy()
    r_ind = est.measure(x_ind, a3, b3)
    print("Independent (expected: II ~ 0, A is sufficient)")
    print(
        f"  I(X;A) = {r_ind.i_xa_bits:.3f}  "
        f"I(X;B) = {r_ind.i_xb_bits:.3f}  "
        f"I(X;A,B) = {r_ind.i_xab_bits:.3f}  "
        f"II = {r_ind.ii_bits:+.3f}"
    )

    # Continuous KSG demo: noisy XOR with continuous noise
    a4 = rng.standard_normal(n)
    b4 = rng.standard_normal(n)
    x_cont = np.sign(a4 * b4) + 0.1 * rng.standard_normal(n)
    est_c = SynergyEstimator(method="ksg", k=4)
    r_cont = est_c.measure(x_cont, a4, b4)
    print("Continuous interaction (X = sign(A*B) + noise; expected II > 0)")
    print(
        f"  I(X;A) = {r_cont.i_xa_bits:.3f}  "
        f"I(X;B) = {r_cont.i_xb_bits:.3f}  "
        f"I(X;A,B) = {r_cont.i_xab_bits:.3f}  "
        f"II = {r_cont.ii_bits:+.3f}"
    )


if __name__ == "__main__":
    _self_demo()
