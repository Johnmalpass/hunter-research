"""Research subsystem.

Modules:
    synergy           interaction information / synergy estimator (S in bits)
    regime_synergy    regime-conditional partial information decomposition
    hunter_bridge     applies the synergy estimator to the HUNTER corpus
    regime            macro regime detector (4 states, soft probabilities)
    predicates        L1+L2 building blocks for compiled mechanisms
    mechanism         Mechanism / Signal base classes + registry
    mechanisms/       compiled mechanisms (one file per thesis)
    coalition         multi-mechanism voting, track-record-weighted
    ledger            mechanism signal/outcome ledger (self-aware substrate)
    backtest          historical evaluation harness for any Mechanism
    compile           LLM mechanism compiler (thesis text -> Python file)
"""

from quant.research.synergy import (
    SynergyEstimator,
    discrete_mi,
    interaction_information,
    ksg_mi,
)

__all__ = [
    "interaction_information",
    "discrete_mi",
    "ksg_mi",
    "SynergyEstimator",
]
