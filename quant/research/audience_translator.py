"""Audience Translator — produce audience-specific articulations of any thesis.

The translation hypothesis (see docstrings of dialect_kl + the regime-conditional
synergy module) says HUNTER's edge depends on TWO multiplicative factors:

  1. Translation quality IN: reading silo dialects into a unified internal form
  2. Translation quality OUT: rendering the internal form into the dialect of
     the audience that can ACT on it.

This module handles the OUT side. Given a HUNTER thesis, it produces N
audience-specific translations — Substack post, SSRN paper, sell-side memo,
Treasury brief, Twitter thread. Each translation has its own conventions
(length, voice, structure, vocabulary) and its own audience that can act on it.

When the TRADER's coalition vote on a thesis crosses threshold, the audience
translator drafts all N translations and stores them as inquiries for John
to review and publish. **Output latency from coalition-decision to
publication-ready text drops from days to minutes.**

Integration with seam network
==============================

Every translation pair HUNTER produces (thesis_internal_form ↔ audience_output)
becomes a documented seam. Over time these seams accumulate into a translation
graph that allows zero-shot translation of NEW theses without LLM calls.

Cost model
==========

One Opus call per (thesis, audience) pair. ~$0.30 per translation, 5
translations per thesis = $1.50/thesis. The TRADER drafts these only when
the coalition vote exceeds a threshold (default: net_confidence >= 0.6 and
total_size >= 0.02 NAV), so we don't burn budget on every minor signal.

Dry-run by default. The CLI enforces --live to actually call the API.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "claude-opus-4-7"


@dataclass
class AudienceProfile:
    """Specification of one audience's expected format and conventions."""

    name: str
    description: str
    target_length_words: int
    voice: str  # short directive: "narrative-first-person", "academic-formal", ...
    structural_template: str
    expected_terms: list[str] = field(default_factory=list)
    forbidden_terms: list[str] = field(default_factory=list)
    audience_decision_criteria: str = ""

    def expected_term_str(self) -> str:
        if not self.expected_terms:
            return "(no specific terminology required)"
        return ", ".join(self.expected_terms)


# ============================================================
# Five default audience profiles
# ============================================================

DEFAULT_PROFILES: dict[str, AudienceProfile] = {
    "substack": AudienceProfile(
        name="substack",
        description=(
            "General-public Substack post. Narrative voice, conversational, "
            "concrete vignettes. Personal stakes. Markdown formatted."
        ),
        target_length_words=1100,
        voice="narrative-first-person",
        structural_template=(
            "# {hook_title}\n\n"
            "*{subtitle}*\n\n"
            "## The setup\n\n[concrete vignette setting up the puzzle]\n\n"
            "## What the data says\n\n[the cross-silo composition explained]\n\n"
            "## Why it matters now\n\n[catalyst, timing, why act]\n\n"
            "## The honest caveats\n\n[limitations, alternative explanations]\n\n"
            "## What I'm doing about it\n\n[your position or recommendation]"
        ),
        expected_terms=["mispriced", "cross-silo", "the data", "I noticed"],
        forbidden_terms=[
            "synergistic information", "partial information decomposition",
            "Markov", "eigenvalue", "Kolmogorov",
        ],
        audience_decision_criteria=(
            "Reader subscribes, shares, or replies with criticism."
        ),
    ),
    "ssrn": AudienceProfile(
        name="ssrn",
        description=(
            "Working-paper format for SSRN. Formal academic tone, abstract "
            "+ introduction + methods + results + discussion + references. "
            "No first person; passive voice acceptable."
        ),
        target_length_words=4000,
        voice="academic-formal",
        structural_template=(
            "# {title}\n\n## Abstract\n\n[200 words]\n\n"
            "## 1. Introduction\n\n## 2. Background and prior work\n\n"
            "## 3. Methodology\n\n## 4. Results\n\n## 5. Discussion\n\n"
            "## 6. Limitations\n\n## 7. Conclusion\n\n## References"
        ),
        expected_terms=[
            "we", "this paper", "our analysis", "we find", "we argue",
            "p < 0.05", "the results suggest",
        ],
        forbidden_terms=["I", "you", "amazing", "huge"],
        audience_decision_criteria=(
            "Reviewer accepts the paper or cites it. Citation count is the metric."
        ),
    ),
    "sell_side": AudienceProfile(
        name="sell_side",
        description=(
            "Institutional sell-side memo. Bottom-Line-Up-Front structure. "
            "Trade idea, catalyst, horizon, position size, key risks."
        ),
        target_length_words=600,
        voice="punchy-direct-imperative",
        structural_template=(
            "**TRADE IDEA: {asset} {direction}**\n\n"
            "**Bottom line:** [one sentence]\n\n"
            "**Catalyst:** [what fires the move + when]\n\n"
            "**Horizon:** [holding period]\n\n"
            "**Mechanism:** [3-5 bullet points of the cross-silo logic]\n\n"
            "**Position sizing:** [recommended % of portfolio]\n\n"
            "**Key risks:** [3 specific things that would invalidate]\n\n"
            "**Stop:** [explicit stop-loss level]"
        ),
        expected_terms=["trade", "catalyst", "horizon", "stop", "tickers"],
        forbidden_terms=["I noticed", "interesting", "perhaps"],
        audience_decision_criteria=(
            "PM puts on the trade in the next 24 hours."
        ),
    ),
    "treasury": AudienceProfile(
        name="treasury",
        description=(
            "Federal Reserve / Treasury / regulator policy brief. Risk-channel "
            "framing. Quantitative impact estimates. Policy options matrix."
        ),
        target_length_words=2500,
        voice="policy-formal-quantitative",
        structural_template=(
            "# Policy Brief: {title}\n\n"
            "## Executive summary\n\n[3 bullet points]\n\n"
            "## Risk channels identified\n\n## Magnitude estimates\n\n"
            "## Policy options\n\n## Recommended monitoring\n\n"
            "## Coordination requirements\n\n## References"
        ),
        expected_terms=[
            "transmission", "systemic", "macroprudential", "regulatory",
            "the Federal Reserve", "Treasury", "stability",
        ],
        forbidden_terms=["amazing", "huge", "tons of", "I think"],
        audience_decision_criteria=(
            "Policymaker references the brief in next FOMC / NAIC / FSB meeting."
        ),
    ),
    "twitter": AudienceProfile(
        name="twitter",
        description=(
            "Compressed thread for X. Hook tweet with strong claim, then 5-10 "
            "follow-up tweets each <= 280 chars. Each tweet self-contained."
        ),
        target_length_words=280,
        voice="compressed-direct",
        structural_template=(
            "Tweet 1 (HOOK, <= 280 chars): [arresting one-liner + claim]\n\n"
            "Tweet 2: [the puzzle]\n"
            "Tweet 3: [the data]\n"
            "Tweet 4: [the cross-silo composition]\n"
            "Tweet 5: [the catalyst]\n"
            "Tweet 6: [the honest caveat]\n"
            "Tweet 7: [the call to action / link]\n"
        ),
        expected_terms=["🧵"],
        forbidden_terms=["abstract", "we conclude"],
        audience_decision_criteria=(
            "Thread gets > 100 retweets or one mention from a known account."
        ),
    ),
}


# ============================================================
# Translation primitives
# ============================================================

@dataclass
class TranslatedThesis:
    thesis_id: str
    audience: str
    translated_text: str
    confidence: float  # how well preserved (0=garbled, 1=perfect)
    cost_usd: float
    word_count: int
    created_at: datetime
    dry_run: bool

    def to_dict(self) -> dict:
        return {
            "thesis_id": self.thesis_id,
            "audience": self.audience,
            "translated_text": self.translated_text,
            "confidence": round(self.confidence, 3),
            "cost_usd": round(self.cost_usd, 4),
            "word_count": self.word_count,
            "created_at": self.created_at.isoformat(),
            "dry_run": self.dry_run,
        }


def build_translation_prompt(
    thesis_text: str,
    profile: AudienceProfile,
    extra_context: Optional[str] = None,
) -> str:
    parts = [
        f"You are translating a HUNTER cross-silo thesis into the format of: {profile.name}",
        f"DESCRIPTION OF THE TARGET AUDIENCE: {profile.description}",
        f"VOICE: {profile.voice}",
        f"TARGET LENGTH: {profile.target_length_words} words.",
        f"EXPECTED TERMINOLOGY: {profile.expected_term_str()}",
        f"AVOID: {', '.join(profile.forbidden_terms) if profile.forbidden_terms else 'no specific bans'}",
        f"WHAT THE READER WILL DO IF SUCCESSFUL: {profile.audience_decision_criteria}",
        "",
        "STRUCTURAL TEMPLATE TO FOLLOW (adapt content, keep structure):",
        profile.structural_template,
        "",
        "THESIS TO TRANSLATE:",
        thesis_text,
    ]
    if extra_context:
        parts.append("")
        parts.append("ADDITIONAL CONTEXT:")
        parts.append(extra_context)
    parts.append("")
    parts.append(
        "OUTPUT: produce ONLY the translated text. No preamble, no commentary, "
        "no markdown fences. Honour the voice, length, and template."
    )
    return "\n".join(parts)


def translate_for_audience(
    thesis_text: str,
    audience: str,
    *,
    thesis_id: str = "untitled",
    profile: Optional[AudienceProfile] = None,
    dry_run: bool = True,
    model: str = DEFAULT_MODEL,
    extra_context: Optional[str] = None,
) -> TranslatedThesis:
    """Translate one thesis into one audience format.

    Default is dry-run (returns the prompt that WOULD be sent without spending
    API budget). Pass dry_run=False to call Anthropic.
    """
    if profile is None:
        if audience not in DEFAULT_PROFILES:
            raise ValueError(
                f"unknown audience '{audience}'; choose from {list(DEFAULT_PROFILES)}"
            )
        profile = DEFAULT_PROFILES[audience]

    prompt = build_translation_prompt(thesis_text, profile, extra_context)
    now = datetime.now(timezone.utc)

    if dry_run:
        return TranslatedThesis(
            thesis_id=thesis_id,
            audience=audience,
            translated_text=f"[DRY-RUN]\n\n{prompt}",
            confidence=0.0,
            cost_usd=0.0,
            word_count=len(prompt.split()),
            created_at=now,
            dry_run=True,
        )

    # Live API call
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic package not installed; pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set in environment")

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    in_tok = msg.usage.input_tokens
    out_tok = msg.usage.output_tokens
    cost = (in_tok * 15 + out_tok * 75) / 1_000_000.0  # Opus 4.x rates
    return TranslatedThesis(
        thesis_id=thesis_id,
        audience=audience,
        translated_text=text,
        confidence=0.85,  # placeholder; replace with embedding-similarity check later
        cost_usd=cost,
        word_count=len(text.split()),
        created_at=now,
        dry_run=False,
    )


def translate_for_all_audiences(
    thesis_text: str,
    *,
    thesis_id: str = "untitled",
    audiences: Optional[list[str]] = None,
    dry_run: bool = True,
    extra_context: Optional[str] = None,
) -> dict[str, TranslatedThesis]:
    """Produce one translation per audience. Returns {audience_name: TranslatedThesis}."""
    audiences = audiences or list(DEFAULT_PROFILES.keys())
    out: dict[str, TranslatedThesis] = {}
    for aud in audiences:
        out[aud] = translate_for_audience(
            thesis_text,
            aud,
            thesis_id=thesis_id,
            dry_run=dry_run,
            extra_context=extra_context,
        )
    return out


# ============================================================
# Persistence: store translated drafts as inquiries
# ============================================================

def stage_translations_as_inquiries(
    translations: dict[str, TranslatedThesis],
    thesis_id: str,
    *,
    db_path: Optional[Path | str] = None,
) -> list[int]:
    """Store each translated draft as an inquiry the operator reviews + publishes.

    Returns the list of inquiry ids. Each inquiry is type=review, urgency=medium
    by default (can be customised per audience).
    """
    from quant.agents.inquiry import open_inquiry

    ids: list[int] = []
    audience_to_urgency = {
        "twitter": "high",         # fastest decay; publish within hours
        "substack": "medium",
        "sell_side": "high",       # institutional readers expect speed
        "treasury": "low",         # policy timing slow
        "ssrn": "low",             # academic timing very slow
    }
    for aud, t in translations.items():
        urgency = audience_to_urgency.get(aud, "medium")
        body = (
            f"Audience-translated draft for thesis '{thesis_id}' "
            f"(audience: {aud}). Review and publish if approved.\n\n"
            f"--- TRANSLATED DRAFT ({t.word_count} words) ---\n"
            f"{t.translated_text[:500]}{'...' if len(t.translated_text) > 500 else ''}"
        )
        ids.append(
            open_inquiry(
                inquiry_type="review",
                urgency=urgency,
                body=body,
                options=["publish", "edit_first", "skip", "wait"],
                related_files=f"audience_translations/{thesis_id}/{aud}",
                db_path=db_path,
            )
        )
    return ids
