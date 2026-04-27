"""LLM mechanism compiler — turn HUNTER theses into executable Python.

Inputs: a HUNTER thesis (text + metadata).
Output: a `quant/research/mechanisms/thesis_<id>.py` file containing a
Mechanism subclass that conforms to the predicate DSL when possible and
falls back to free-form Python when needed.

Workflow:
  1. Build a prompt that includes:
       - 1-2 in-context examples of valid Mechanism subclasses
       - the catalog of available predicates (the DSL)
       - the catalog of available data fields per adapter
       - the constraints (signature, no LLM at runtime, must return [] on missing data)
       - the thesis text + metadata
  2. Call Anthropic Opus 4.7
  3. Parse the response (strip markdown fences if present)
  4. Validate via AST (must have @register, must have evaluate, etc.)
  5. On validation failure: retry up to N times with the error message included
  6. Write the validated source to disk
  7. Importing the file auto-registers the mechanism in the registry

Cost model: ~10K prompt tokens + ~3K output tokens per thesis on Opus 4.7
~$0.50 per thesis. 18 diamond theses ~= $9.

Usage from CLI:
    python -m quant compile --dry-run --thesis-id 328  --text "<thesis>"
    python -m quant compile --live    --thesis-id 328  --text "<thesis>"
"""
from __future__ import annotations

import ast
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DEFAULT_MODEL = "claude-opus-4-7"

PROMPT_HEADER = """You are compiling a HUNTER cross-silo financial thesis into executable Python.

OUTPUT FORMAT
=============

Output ONLY the contents of a single Python source file. No commentary.
No markdown fences. The file must:

  1. Import from `quant.data.base`, `quant.research.mechanism`, optionally
     `quant.research.predicates`.
  2. Define exactly one `@register("<thesis_id>")`-decorated subclass of
     `Mechanism` (a dataclass with `kw_only=True`).
  3. Provide class-level dataclass fields with defaults for every inherited
     Mechanism field (thesis_id, name, universe, requirements,
     holding_period_days, direction, description).
  4. Provide a `__post_init__` that fills in `universe`, `requirements`,
     and default `params`.
  5. Provide an `evaluate(self, state, asof) -> list[Signal]` method that
     reads from MarketState, applies the thesis logic, and returns zero
     or more `Signal`s.

EXAMPLE OF A VALID COMPILED MECHANISM
=====================================

```python
\"\"\"Mechanism: example credit-spread blowout.

When BAA-Treasury credit spread z-scores above 2 in a flat-curve regime,
short a basket of high-yield-issuer equities for 60 days.
\"\"\"
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta

from quant.data.base import MarketState
from quant.research.mechanism import (
    Mechanism, MechanismRequirement, Signal, register,
)
from quant.research.predicates import (
    And, RegimePredicate, ZScorePredicate,
)


@register("example_credit_blowout")
@dataclass(kw_only=True)
class ExampleCreditBlowout(Mechanism):
    thesis_id: str = "example_credit_blowout"
    name: str = "Credit-spread blowout in flat-curve regime"
    universe: list[str] | None = None
    requirements: list[MechanismRequirement] | None = None
    holding_period_days: int = 60
    direction: str = "short"
    description: str = "Short HY issuers when BAA10Y z>2 and regime late_cycle/risk_off."

    def __post_init__(self) -> None:
        self.universe = ["HYG", "JNK"]  # high-yield ETFs as a basket proxy
        self.requirements = [
            MechanismRequirement("BAA10Y", "value", "fred",
                                 note="python -m quant ingest fred --series BAA10Y"),
            MechanismRequirement("DGS10", "value", "fred",
                                 note="python -m quant ingest fred --series DGS10"),
            MechanismRequirement("DGS2", "value", "fred",
                                 note="python -m quant ingest fred --series DGS2"),
        ]
        self.params.setdefault("z_threshold", 2.0)
        self.params.setdefault("size_per_name_pct", 0.02)

    def evaluate(self, state: MarketState, asof: datetime) -> list[Signal]:
        gate = And(
            ZScorePredicate("BAA10Y", "value",
                            threshold=self.params["z_threshold"], window_days=180),
            RegimePredicate(allowed_regimes=["late_cycle", "risk_off", "crisis"],
                            min_probability=0.4),
        )
        result = gate(state, asof)
        if result is None or not result.fired:
            return []
        return [
            Signal(
                asset=ticker,
                direction="short",
                size_pct=float(self.params["size_per_name_pct"]),
                confidence=min(1.0, result.magnitude / 4.0),
                holding_period_days=self.holding_period_days,
                rationale=f"BAA10Y z-score and regime gate fired (mag={result.magnitude:.2f})",
                asof=asof,
                metadata={"gate_evidence": result.evidence},
            )
            for ticker in self.universe
        ]
```

AVAILABLE PREDICATES (Level 1 — prefer these when applicable)
============================================================

  ThresholdPredicate(asset_id, field, threshold, op=">=" | ">" | "<=" | "<" | "==")
  ZScorePredicate(asset_id, field, threshold, window_days=90, min_obs=30)
  SpreadPredicate(a_asset, a_field, b_asset, b_field, threshold, scale=1.0, op=">=")
  RegimePredicate(allowed_regimes=[...], min_probability=0.5)
  WithinDaysOfPredicate(calendar_days=[(month, day), ...], max_days=N)

COMBINATORS (Level 2)
=====================

  And(p1, p2, ...)   Or(p1, p2, ...)   Not(p)

  And short-circuits to None if any child returns None (missing data).

AVAILABLE DATA FIELDS
=====================

FRED (free, ingested by quant.data.adapters.fred):
  asset_id one of: DGS10, DGS2, DGS1, DGS5, DGS30, BAA10Y, BAMLC0A0CM,
  CPIAUCSL, CPILFESL, INDPRO, UNRATE, PAYEMS, WALCL, M2SL,
  MORTGAGE30US, CSUSHPISA, DTWEXBGS, RRPONTSYD
  field = "value"

SEC EDGAR (free):
  asset_id = 10-digit zero-padded CIK string (e.g. "0000320193" for AAPL)
  field examples = "filing_10_k", "filing_10_q", "filing_8_k", "filing_4", "filing_13_f"

FAERS (free, FDA adverse-event reports):
  asset_id = drug name UPPERCASE
  field = "faers_reports_count_1d"

Polygon (paid, equity OHLCV):
  asset_id = ticker UPPERCASE
  field = "price_close" | "price_open" | "price_high" | "price_low" | "volume" | "vwap"

Tiingo (paid, news):
  asset_id = ticker UPPERCASE
  field = "news_count_1d" | "news_sentiment_avg_1d"

GDELT (free, global news events):
  asset_id = sanitised query string (lowercase, underscores)
  field = "articles_count_1d" | "mean_tone_1d"

CONSTRAINTS
===========

- Use ONLY the data fields listed above. If the thesis needs data we
  don't have, list that as a MechanismRequirement with a clear ingestion
  command in the `note` field.
- Prefer predicates from the DSL over hand-rolled logic when applicable.
- evaluate() must NOT call any LLM. It is pure Python that reads
  MarketState and emits Signals.
- evaluate() must return [] when required data is missing — never raise.
- All numeric thresholds must live in self.params with sensible defaults.
- The thesis_id parameter passed to @register MUST match the class
  attribute thesis_id and the filename.

THESIS TO COMPILE
=================
"""


@dataclass
class CompileResult:
    status: str  # "ok" | "validation_failed" | "api_error" | "dry_run"
    code: Optional[str] = None
    prompt: Optional[str] = None
    validation_message: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    written_to: Optional[Path] = None
    error: Optional[str] = None


def build_prompt(
    thesis_id: str,
    thesis_title: str,
    thesis_score: Optional[float],
    silos: list[str],
    thesis_text: str,
    extra_context: Optional[str] = None,
) -> str:
    body = (
        f"thesis_id: {thesis_id}\n"
        f"thesis_title: {thesis_title}\n"
        f"thesis_score: {thesis_score if thesis_score is not None else 'unknown'}\n"
        f"silos: {', '.join(silos)}\n\n"
        f"DESCRIPTION:\n{thesis_text}\n"
    )
    if extra_context:
        body += f"\nADDITIONAL CONTEXT:\n{extra_context}\n"
    body += "\nOutput the Python source file ONLY. No commentary.\n"
    return PROMPT_HEADER + body


def _strip_code_fences(text: str) -> str:
    """If the model wrapped the code in ```python ... ``` strip the fences."""
    text = text.strip()
    fence = re.compile(r"^```(?:python)?\n", re.IGNORECASE)
    m = fence.match(text)
    if m:
        text = text[m.end():]
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text


def validate_compiled_code(code: str, expected_thesis_id: str) -> tuple[bool, str]:
    """AST-validate the generated source.

    Checks:
      - parses as Python
      - imports from quant.research.mechanism
      - has at least one @register-decorated class
      - that class has an evaluate method
      - the registered thesis_id matches expected_thesis_id
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"

    has_mech_import = False
    register_decorators_with_id: list[str] = []
    has_evaluate = False
    found_thesis_class = False

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "quant.research.mechanism":
                names = {a.name for a in node.names}
                if "Mechanism" in names and "register" in names:
                    has_mech_import = True
        if isinstance(node, ast.ClassDef):
            for dec in node.decorator_list:
                if (
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Name)
                    and dec.func.id == "register"
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                ):
                    register_decorators_with_id.append(str(dec.args[0].value))
                    found_thesis_class = True
            if found_thesis_class:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "evaluate":
                        has_evaluate = True

    if not has_mech_import:
        return False, "missing `from quant.research.mechanism import Mechanism, register, ...`"
    if not register_decorators_with_id:
        return False, "no @register-decorated class found"
    if not has_evaluate:
        return False, "the @register-decorated class is missing an evaluate() method"
    if expected_thesis_id not in register_decorators_with_id:
        return False, (
            f"expected @register('{expected_thesis_id}') but got "
            f"{register_decorators_with_id}"
        )
    return True, "ok"


def call_llm(
    prompt: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> dict:
    """Call the Anthropic API. Returns dict with 'text', usage stats, errors."""
    try:
        import anthropic
    except ImportError:
        return {"error": "anthropic package not installed; pip install anthropic"}

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"error": "ANTHROPIC_API_KEY not set in .env"}

    try:
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text"))
        return {
            "text": text,
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
            # Opus 4.x rates: $15/Mtok in, $75/Mtok out (approximate)
            "cost_usd": (
                msg.usage.input_tokens * 15 + msg.usage.output_tokens * 75
            ) / 1_000_000.0,
        }
    except Exception as e:
        return {"error": str(e)}


def write_compiled_mechanism(
    thesis_id: str,
    code: str,
    mechanisms_dir: Optional[Path] = None,
    overwrite: bool = False,
) -> Path:
    """Write the validated source to quant/research/mechanisms/thesis_<id>.py."""
    if mechanisms_dir is None:
        mechanisms_dir = Path(__file__).resolve().parent / "mechanisms"
    mechanisms_dir.mkdir(parents=True, exist_ok=True)

    stem = thesis_id if thesis_id.startswith("thesis_") else f"thesis_{thesis_id}"
    safe_stem = re.sub(r"[^a-z0-9_]", "_", stem.lower())
    path = mechanisms_dir / f"{safe_stem}.py"

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} exists; pass overwrite=True to replace"
        )
    path.write_text(code)
    return path


def compile_thesis(
    thesis_id: str,
    thesis_title: str,
    thesis_text: str,
    silos: Optional[list[str]] = None,
    thesis_score: Optional[float] = None,
    *,
    extra_context: Optional[str] = None,
    dry_run: bool = True,
    model: str = DEFAULT_MODEL,
    max_retries: int = 1,
    overwrite: bool = False,
    mechanisms_dir: Optional[Path] = None,
) -> CompileResult:
    """End-to-end compile: prompt -> LLM -> parse -> validate -> write."""
    silos = silos or []
    prompt = build_prompt(
        thesis_id, thesis_title, thesis_score, silos, thesis_text, extra_context
    )

    if dry_run:
        return CompileResult(
            status="dry_run",
            prompt=prompt,
            validation_message="(no API call made)",
        )

    last_err: Optional[str] = None
    for attempt in range(max_retries + 1):
        prompt_for_attempt = prompt
        if last_err and attempt > 0:
            prompt_for_attempt += (
                f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {last_err}\n"
                f"Fix the issue and produce a valid file.\n"
            )
        api = call_llm(prompt_for_attempt, model=model)
        if "error" in api:
            return CompileResult(
                status="api_error",
                prompt=prompt,
                error=api["error"],
            )

        code = _strip_code_fences(api["text"])
        valid, msg = validate_compiled_code(code, thesis_id)
        if valid:
            try:
                path = write_compiled_mechanism(
                    thesis_id, code, mechanisms_dir=mechanisms_dir, overwrite=overwrite
                )
            except FileExistsError as e:
                return CompileResult(
                    status="api_error",
                    code=code,
                    prompt=prompt,
                    input_tokens=api.get("input_tokens"),
                    output_tokens=api.get("output_tokens"),
                    cost_usd=api.get("cost_usd"),
                    error=str(e),
                )
            return CompileResult(
                status="ok",
                code=code,
                prompt=prompt,
                input_tokens=api.get("input_tokens"),
                output_tokens=api.get("output_tokens"),
                cost_usd=api.get("cost_usd"),
                validation_message=msg,
                written_to=path,
            )
        last_err = msg

    return CompileResult(
        status="validation_failed",
        code=code,
        prompt=prompt,
        validation_message=last_err,
        input_tokens=api.get("input_tokens"),
        output_tokens=api.get("output_tokens"),
        cost_usd=api.get("cost_usd"),
    )
