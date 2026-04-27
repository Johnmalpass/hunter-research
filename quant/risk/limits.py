"""Hard risk limits enforced in code.

The CONSCIENCE agent gets advisory; this module is law. Every order proposed
by the TRADER passes through `RiskLimits.check_*` before it ever leaves the
process. A failed check raises `RiskHalted`, which the supervisor logs and
human-confirms before resuming.

Three numbers govern everything:

  max_position_pct      max single position as % of net liquidation value
  max_drawdown_pct      portfolio peak-to-trough drawdown that triggers full exit
  max_daily_loss_pct    single-day P&L drop that triggers a 24h halt

Defaults below are conservative for a one-operator prop shop. Override per
account in `RiskLimits(...)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NamedTuple


class RiskHalted(Exception):
    """Raised when a hard cap is breached. Stops trading until human confirms."""


class RiskCheck(NamedTuple):
    ok: bool
    reason: str


@dataclass
class RiskLimits:
    """All values are decimal fractions (0.05 = 5%). Edit per-account."""

    max_position_pct: float = 0.05
    max_drawdown_pct: float = 0.08
    max_daily_loss_pct: float = 0.02
    max_gross_leverage: float = 1.50  # 1.0 = unleveraged; >1 = margin
    max_concentration_per_silo_pct: float = 0.20
    max_correlated_position_pct: float = 0.15  # for clustered theses (e.g. multiple CMBS shorts)

    def check_position_size(
        self,
        position_dollars: float,
        nav_dollars: float,
    ) -> RiskCheck:
        if nav_dollars <= 0:
            return RiskCheck(False, "nav <= 0; halt all new positions")
        pct = abs(position_dollars) / nav_dollars
        if pct > self.max_position_pct:
            return RiskCheck(
                False,
                f"position {pct:.2%} > cap {self.max_position_pct:.2%}",
            )
        return RiskCheck(True, "")

    def check_drawdown(
        self,
        current_nav: float,
        peak_nav: float,
    ) -> RiskCheck:
        if peak_nav <= 0:
            return RiskCheck(True, "no peak yet")
        dd = (peak_nav - current_nav) / peak_nav
        if dd > self.max_drawdown_pct:
            return RiskCheck(
                False,
                f"drawdown {dd:.2%} > cap {self.max_drawdown_pct:.2%} - exit all",
            )
        return RiskCheck(True, f"drawdown {dd:.2%}")

    def check_daily_loss(
        self,
        nav_today: float,
        nav_yesterday: float,
    ) -> RiskCheck:
        if nav_yesterday <= 0:
            return RiskCheck(True, "no yesterday nav")
        loss = (nav_yesterday - nav_today) / nav_yesterday
        if loss > self.max_daily_loss_pct:
            return RiskCheck(
                False,
                f"daily loss {loss:.2%} > cap {self.max_daily_loss_pct:.2%} - 24h halt",
            )
        return RiskCheck(True, f"daily P&L {-loss:+.2%}")

    def check_gross_leverage(
        self,
        gross_exposure: float,
        nav_dollars: float,
    ) -> RiskCheck:
        if nav_dollars <= 0:
            return RiskCheck(False, "nav <= 0")
        lev = gross_exposure / nav_dollars
        if lev > self.max_gross_leverage:
            return RiskCheck(
                False,
                f"gross leverage {lev:.2f}x > cap {self.max_gross_leverage:.2f}x",
            )
        return RiskCheck(True, f"leverage {lev:.2f}x")

    def check_silo_concentration(
        self,
        silo_exposure_dollars: float,
        nav_dollars: float,
        silo: str,
    ) -> RiskCheck:
        if nav_dollars <= 0:
            return RiskCheck(False, "nav <= 0")
        pct = abs(silo_exposure_dollars) / nav_dollars
        if pct > self.max_concentration_per_silo_pct:
            return RiskCheck(
                False,
                f"silo {silo} concentration {pct:.2%} > "
                f"cap {self.max_concentration_per_silo_pct:.2%}",
            )
        return RiskCheck(True, "")


@dataclass
class RiskState:
    """Mutable run-time state the limits operate on."""

    nav: float
    peak_nav: float
    nav_yesterday: float | None = None
    halted_until: datetime | None = None

    def is_halted(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        return self.halted_until is not None and now < self.halted_until

    def update_nav(self, new_nav: float) -> None:
        self.nav = new_nav
        if new_nav > self.peak_nav:
            self.peak_nav = new_nav
