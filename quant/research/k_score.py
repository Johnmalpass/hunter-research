"""Kolmogorov-Solomonoff K-Score: a description-length proxy for compositional depth.

The intuition
=============

True Kolmogorov complexity K(s) — the length of the shortest program that
produces string s — is uncomputable. But real-world compression algorithms
(LZMA, BZIP2, gzip) produce *approximations* of it. The compressed length of
s is an upper bound on K(s) plus a constant.

For HUNTER theses, we use compression as an estimator of "how much
information is in this thesis." Theses that are genuinely compositional —
that span multiple silos with cross-references and joint structure — have
LONGER compressed forms than the sum of their constituent facts compressed
individually. The DIFFERENCE is a measure of compositional depth.

Formally, define the **K-score** of a thesis t composed of facts f_1,...,f_n:

    K_score(t) = K(t) - sum_i K(f_i)

In approximation:

    K_score(t) ~ |compress(t)| - sum_i |compress(f_i)|

Theses with high K-score are *not* compressible by knowing the constituent
facts — they have genuinely emergent compositional structure. Theses with
low or negative K-score are essentially additive — knowing the facts gives
you the thesis for free.

This is the algorithmic-information-theory complement to:
  - Synergy bits (information theory, partial info decomposition)
  - Demon Index (information thermodynamics, Maxwell-demon bound)

Three independent measures of the same underlying property: how much of
the thesis is structurally cross-silo as opposed to additive single-silo.

Why three lenses
================

Each lens has different failure modes:
  - Synergy bits assumes good estimators of mutual information; small samples
    bias them.
  - Demon Index assumes a calibrated alpha-per-bit constant; before summer
    we use a placeholder.
  - K-score assumes the compression algorithm approximates Kolmogorov
    complexity; it does for natural-language structured text but not for
    arbitrary strings.

When all three agree, the thesis is robustly compositional. When they
disagree, the disagreement points to a specific assumption that's
breaking — useful diagnostic, not a contradiction.

This is, as far as we have searched, the first time algorithmic information
theory has been applied as a compositional-depth estimator on cross-silo
financial theses. It is also computable on natural-language text from the
HUNTER corpus directly — no live API calls, no proprietary data required.
"""
from __future__ import annotations

import lzma
from dataclasses import dataclass
from typing import Sequence


@dataclass
class KScoreResult:
    """Compositional-depth estimate via compression."""

    thesis_text: str
    n_constituent_facts: int
    compressed_thesis_bytes: int
    compressed_facts_bytes_sum: int
    raw_thesis_bytes: int
    k_score_bytes: int
    k_score_normalised: float  # k_score / raw_thesis_bytes, in roughly [-1, 1]
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "n_constituent_facts": self.n_constituent_facts,
            "raw_thesis_bytes": self.raw_thesis_bytes,
            "compressed_thesis_bytes": self.compressed_thesis_bytes,
            "compressed_facts_bytes_sum": self.compressed_facts_bytes_sum,
            "k_score_bytes": self.k_score_bytes,
            "k_score_normalised": round(self.k_score_normalised, 4),
            "interpretation": self.interpretation,
        }


def _compressed_size(text: str, preset: int = 6) -> int:
    """LZMA-compressed byte length of `text`. Returns >= 1.

    `preset` is LZMA compression level (0..9). Higher = better approximation
    of Kolmogorov complexity but slower. Default 6 is a reasonable middle.
    """
    if not text:
        return 1
    payload = text.encode("utf-8")
    compressed = lzma.compress(payload, preset=preset)
    return len(compressed)


def k_score(
    thesis_text: str,
    constituent_facts: Sequence[str],
) -> KScoreResult:
    """Compute the K-score of a thesis vs its constituent facts.

    A thesis whose compressed length is roughly equal to the SUM of its
    facts' compressed lengths has low compositional depth — knowing the
    facts is enough. A thesis whose compressed length materially EXCEEDS the
    sum suggests genuine emergent structure (cross-silo references, joint
    qualifications, conditional logic that the facts alone don't carry).

    A NEGATIVE K-score means the thesis is more compressible than its
    facts — typically because the thesis consolidates redundant content.
    Negative is fine; it just means compositional depth is low.
    """
    if not thesis_text:
        raise ValueError("thesis_text cannot be empty")

    raw = len(thesis_text.encode("utf-8"))
    c_thesis = _compressed_size(thesis_text)
    c_facts_sum = sum(_compressed_size(f) for f in constituent_facts)
    delta = c_thesis - c_facts_sum
    normalised = delta / max(1, raw)

    if normalised > 0.20:
        interp = (
            "high compositional depth: thesis description requires "
            "substantially more bits than its constituent facts; "
            "structurally cross-silo"
        )
    elif normalised > 0.05:
        interp = (
            "moderate compositional depth: thesis encodes meaningful "
            "joint structure beyond its facts"
        )
    elif normalised > -0.10:
        interp = (
            "low compositional depth: thesis is roughly additive over its "
            "facts; consider whether it's genuinely cross-silo"
        )
    else:
        interp = (
            "highly compressible vs facts: the thesis collapses redundancy; "
            "this typically means it's a polished restatement of a single fact"
        )

    return KScoreResult(
        thesis_text=thesis_text,
        n_constituent_facts=len(constituent_facts),
        compressed_thesis_bytes=c_thesis,
        compressed_facts_bytes_sum=c_facts_sum,
        raw_thesis_bytes=raw,
        k_score_bytes=delta,
        k_score_normalised=normalised,
        interpretation=interp,
    )


def normalised_compression_distance(a: str, b: str) -> float:
    """Normalised Compression Distance — a metric in [0, 1] between two texts.

    NCD(a, b) = (C(ab) - min(C(a), C(b))) / max(C(a), C(b))
    where C is compressed length.

    Cilibrasi & Vitanyi 2005. NCD = 0 means perfectly similar; NCD = 1 means
    maximally different. We use this elsewhere to cluster similar HUNTER
    theses and detect duplicates the embedding space might miss.
    """
    if not a or not b:
        return 1.0
    c_a = _compressed_size(a)
    c_b = _compressed_size(b)
    c_ab = _compressed_size(a + " " + b)
    return (c_ab - min(c_a, c_b)) / max(c_a, c_b)
