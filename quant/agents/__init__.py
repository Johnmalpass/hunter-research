"""Agent subsystem.

Modules:
    inquiry      the system asks the operator questions; operator answers daily
    conscience   pre-trade risk veto (rule-based; Opus-backed in v2)
    trader       the orchestrator: state -> regime -> mechs -> coalition ->
                 sizing -> conscience -> ledger

Future:
    auditor    continuous-validation agent (re-checks open positions)
    memory     episodic + semantic memory layer for long-running runs
"""
from quant.agents.conscience import (
    ConscienceVerdict,
    OpenPosition,
    ProposedOrder,
    Verdict,
    review_order,
)
from quant.agents.inquiry import (
    Inquiry,
    answer_inquiry,
    dismiss_inquiry,
    list_open_inquiries,
    open_inquiry,
)
from quant.agents.trader import (
    OrderRecord,
    TradingCycleResult,
    run_cycle,
)

__all__ = [
    # inquiry
    "Inquiry", "answer_inquiry", "dismiss_inquiry",
    "list_open_inquiries", "open_inquiry",
    # conscience
    "ConscienceVerdict", "OpenPosition", "ProposedOrder", "Verdict",
    "review_order",
    # trader
    "OrderRecord", "TradingCycleResult", "run_cycle",
]
