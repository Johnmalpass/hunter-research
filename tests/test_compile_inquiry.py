"""Tests for compile.py and agents/inquiry.py.

The compiler tests are deliberately offline — they exercise prompt building,
fence stripping, AST validation, and the dry-run path. Live API calls are
not invoked from tests (they cost money and are non-deterministic).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from quant.agents.inquiry import (
    Inquiry,
    answer_inquiry,
    dismiss_inquiry,
    get_inquiry,
    list_open_inquiries,
    open_inquiry,
)
from quant.research.compile import (
    _strip_code_fences,
    build_prompt,
    compile_thesis,
    validate_compiled_code,
    write_compiled_mechanism,
)


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    return tmp_path / "test_ci.db"


# ============================================================
# Compile — prompt, fence stripping, validation
# ============================================================

def test_build_prompt_includes_thesis_text():
    p = build_prompt(
        thesis_id="example_x",
        thesis_title="Test thesis",
        thesis_score=78.0,
        silos=["FRED", "EDGAR"],
        thesis_text="When yield curve inverts, short banks.",
    )
    assert "thesis_id: example_x" in p
    assert "Test thesis" in p
    assert "FRED" in p
    assert "When yield curve inverts" in p
    # Must reference the predicate DSL
    assert "ZScorePredicate" in p
    assert "RegimePredicate" in p
    assert "AVAILABLE DATA FIELDS" in p


def test_strip_code_fences_python_fence():
    text = '```python\nprint("hi")\n```'
    assert _strip_code_fences(text) == 'print("hi")'


def test_strip_code_fences_plain_fence():
    text = "```\nprint('hi')\n```"
    assert _strip_code_fences(text) == "print('hi')"


def test_strip_code_fences_no_fence():
    text = "print('hi')"
    assert _strip_code_fences(text) == "print('hi')"


VALID_MECHANISM_SOURCE = '''"""Mechanism: test.

Test description.
"""
from dataclasses import dataclass
from datetime import datetime
from quant.data.base import MarketState
from quant.research.mechanism import (
    Mechanism, MechanismRequirement, Signal, register,
)


@register("test_compile_x")
@dataclass(kw_only=True)
class TestMech(Mechanism):
    thesis_id: str = "test_compile_x"
    name: str = "Test"
    universe: list[str] | None = None
    requirements: list[MechanismRequirement] | None = None
    holding_period_days: int = 30
    direction: str = "short"
    description: str = "test"

    def __post_init__(self) -> None:
        self.universe = ["X"]
        self.requirements = []

    def evaluate(self, state: MarketState, asof: datetime) -> list[Signal]:
        return []
'''


def test_validate_compiled_code_accepts_valid():
    ok, msg = validate_compiled_code(VALID_MECHANISM_SOURCE, "test_compile_x")
    assert ok, msg


def test_validate_compiled_code_rejects_wrong_id():
    ok, msg = validate_compiled_code(VALID_MECHANISM_SOURCE, "different_id")
    assert not ok
    assert "expected" in msg


def test_validate_compiled_code_rejects_no_register():
    bad = VALID_MECHANISM_SOURCE.replace('@register("test_compile_x")\n', "")
    ok, msg = validate_compiled_code(bad, "test_compile_x")
    assert not ok
    assert "register" in msg


def test_validate_compiled_code_rejects_syntax_error():
    bad = "def x(:\n    pass"
    ok, msg = validate_compiled_code(bad, "test_compile_x")
    assert not ok
    assert "syntax" in msg


def test_dry_run_returns_prompt_no_api(tmp_path):
    res = compile_thesis(
        thesis_id="example_dry",
        thesis_title="dry test",
        thesis_text="when X happens, do Y",
        silos=["FRED"],
        dry_run=True,
        mechanisms_dir=tmp_path,
    )
    assert res.status == "dry_run"
    assert res.prompt is not None
    assert "example_dry" in res.prompt
    assert res.code is None
    assert res.input_tokens is None  # no API hit


def test_write_compiled_mechanism(tmp_path):
    target_dir = tmp_path / "mechs"
    path = write_compiled_mechanism(
        "test_xyz", VALID_MECHANISM_SOURCE, mechanisms_dir=target_dir
    )
    assert path.exists()
    assert path.name == "thesis_test_xyz.py"
    assert "test_compile_x" in path.read_text()


def test_write_compiled_mechanism_no_overwrite_by_default(tmp_path):
    target_dir = tmp_path / "mechs"
    write_compiled_mechanism("y", VALID_MECHANISM_SOURCE, mechanisms_dir=target_dir)
    with pytest.raises(FileExistsError):
        write_compiled_mechanism(
            "y", VALID_MECHANISM_SOURCE, mechanisms_dir=target_dir
        )


# ============================================================
# Inquiry queue
# ============================================================

def test_open_and_list_inquiry(tmp_db: Path):
    iid = open_inquiry(
        inquiry_type="decision",
        body="Should we exit ABBV early? FAERS spike has decayed.",
        urgency="high",
        options=["exit", "hold", "scale_down"],
        related_files="quant/research/mechanisms/pharma_adverse_spike.py",
        db_path=tmp_db,
    )
    assert iid > 0
    open_now = list_open_inquiries(db_path=tmp_db)
    assert len(open_now) == 1
    assert open_now[0].inquiry_type == "decision"
    assert open_now[0].urgency == "high"
    assert "exit" in open_now[0].options


def test_inquiries_ordered_by_urgency(tmp_db: Path):
    open_inquiry(inquiry_type="data", body="low one", urgency="low", db_path=tmp_db)
    open_inquiry(
        inquiry_type="decision", body="critical one", urgency="critical", db_path=tmp_db
    )
    open_inquiry(inquiry_type="review", body="medium one", urgency="medium", db_path=tmp_db)
    items = list_open_inquiries(db_path=tmp_db)
    assert [i.urgency for i in items] == ["critical", "medium", "low"]


def test_answer_removes_from_open(tmp_db: Path):
    iid = open_inquiry(
        inquiry_type="decision", body="?", urgency="high", db_path=tmp_db
    )
    answer_inquiry(iid, "exit", db_path=tmp_db)
    open_now = list_open_inquiries(db_path=tmp_db)
    assert len(open_now) == 0
    inq = get_inquiry(iid, db_path=tmp_db)
    assert inq.answer == "exit"
    assert inq.answered_at is not None
    assert not inq.is_open


def test_dismiss_removes_from_open(tmp_db: Path):
    iid = open_inquiry(
        inquiry_type="data", body="?", urgency="medium", db_path=tmp_db
    )
    dismiss_inquiry(iid, db_path=tmp_db)
    open_now = list_open_inquiries(db_path=tmp_db)
    assert len(open_now) == 0
    inq = get_inquiry(iid, db_path=tmp_db)
    assert inq.dismissed
    assert inq.answer is None


def test_invalid_inquiry_type_rejected(tmp_db: Path):
    with pytest.raises(ValueError):
        open_inquiry(inquiry_type="bogus", body="?", db_path=tmp_db)


def test_invalid_urgency_rejected(tmp_db: Path):
    with pytest.raises(ValueError):
        open_inquiry(
            inquiry_type="data", body="?", urgency="urgent_now", db_path=tmp_db
        )
