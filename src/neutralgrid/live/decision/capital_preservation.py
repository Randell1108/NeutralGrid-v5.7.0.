"""Source-bound invested-capital observations, never action authorization.

The floor is the explicitly reported invested capital in an exact-time equity
fact. It is not peak equity or capital plus a displayed PnL estimate. A passing
comparison describes one observation; independent freshness, liquidation,
drawdown, execution and training gates still apply.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Context, Decimal, DecimalException, localcontext
from pathlib import Path
from typing import Any, Literal

from neutralgrid.live.decision.capital_policy import policy_summary
from neutralgrid.live.decision.exact_bot_ledger import PrivateFact, verify_fact
from neutralgrid.live.decision.state_action_reward import (
    ContractModel, SarError, SourceRef, State, decimal, utc,
)

__all__ = ["CapitalAssessment", "assess_current_capital", "policy_summary"]


class CapitalAssessment(ContractModel):
    primary_objective: Literal["preserve_invested_capital"] = "preserve_invested_capital"
    reference: Literal["invested_capital"] = "invested_capital"
    scope: Literal["current_observation_only"] = "current_observation_only"
    status: Literal["missing", "breached", "observed_preserved"]
    diagnostics: tuple[str, ...]
    bot_key: str | None = None
    account_scope: str | None = None
    observed_at: datetime | None = None
    invested_capital_usdt: Decimal | None = None
    equity_usdt: Decimal | None = None
    headroom_usdt: Decimal | None = None
    source: SourceRef | None = None
    future_preservation_guaranteed: Literal[False] = False
    execution_authorized: Literal[False] = False
    training_eligible: Literal[False] = False

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def assess_current_capital(
    state: State,
    equity_fact: PrivateFact | None,
    *,
    live_root: Path,
    asof: datetime | None = None,
    expected_account_scope: str | None = None,
) -> CapitalAssessment:
    """Assess one exact native observation, gracefully rejecting bad evidence.

Uses the same identity, timing and accounting alignment as native_state's
equity admission. verify_fact uses a bounded source read and exact source-field
binding; that integrity check does not authenticate an arbitrary producer.
The caller must use a reviewed source producer. No source is written, and no
wall-clock freshness threshold or unobserved path guarantee is inferred.
"""
    def missing(reason: str) -> CapitalAssessment:
        return CapitalAssessment(status="missing", diagnostics=(reason,))

    if equity_fact is None:
        return missing("capital_floor_evidence_missing")
    try:
        checked_state = State.model_validate(state.model_dump(mode="python"))
        at = utc(asof if asof is not None else checked_state.available_at)
        fact = PrivateFact.model_validate(equity_fact.model_dump(mode="python"))
    except (ValueError, TypeError, SarError):
        return missing("capital_floor_input_invalid")
    if fact.kind != "equity":
        return missing("capital_floor_equity_fact_required")
    if fact.bot != checked_state.bot:
        return missing("capital_floor_identity_mismatch")
    if expected_account_scope is not None and fact.account_scope != expected_account_scope:
        return missing("capital_floor_account_mismatch")
    if fact.occurred_at != checked_state.captured_at:
        return missing("capital_floor_time_mismatch")
    if not fact.available_at <= checked_state.available_at <= at:
        return missing("capital_floor_unavailable")
    if checked_state.capital_usdt is None:
        return missing("capital_floor_unknown")
    floor = decimal(fact.fields["invested_capital_usdt"])
    equity = decimal(fact.fields["equity_usdt"])
    if (floor != checked_state.capital_usdt or
            decimal(fact.fields["unrealized_usdt"]) != checked_state.unrealized_usdt or
            (checked_state.equity_usdt is not None and equity != checked_state.equity_usdt)):
        return missing("capital_floor_accounting_mismatch")
    try:
        verify_fact(fact, live_root=live_root)
    except (OSError, ValueError, TypeError, KeyError, IndexError, RecursionError, SarError):
        return missing("capital_floor_source_invalid")
    preserved = equity >= floor
    try:
        # Source-bound finite numbers can still exceed Decimal's arithmetic
        # exponent range. Do not inherit a caller's precision or trap settings.
        with localcontext(Context(prec=50)):
            headroom = equity - floor
    except DecimalException:
        return missing("capital_floor_arithmetic_unavailable")
    return CapitalAssessment(
        status="observed_preserved" if preserved else "breached",
        diagnostics=("capital_floor_observed_preserved" if preserved else "capital_floor_breached",
                     "future_capital_preservation_unproven", "independent_risk_limits_not_assessed"),
        bot_key=checked_state.bot.key, account_scope=fact.account_scope,
        observed_at=checked_state.captured_at, invested_capital_usdt=floor,
        equity_usdt=equity, headroom_usdt=headroom, source=fact.source,
    )
