"""Common research envelope; target semantics and native evidence stay separate.

Validation establishes structural consistency and optional source integrity, not
the truth of a producer's coverage assertion. A native adapter must independently
validate that assertion before training admission. This module never grants
training eligibility, promotes a model, or influences a live verdict.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from neutralgrid.live.decision.state_action_reward import (
    BotIdentity, ContractModel, SourceRef, decimal, file_hash, fingerprint, utc,
)


SCHEMA = "neutralgrid_forecast_evidence_contract_v1"
TARGET_PCT = Decimal("3")
DEPLOYMENT_HORIZON = timedelta(hours=7)


class TimedSource(ContractModel):
    """One immutable native record; availability is distinct from event time."""

    record_id: str = Field(min_length=1, max_length=256)
    occurred_at: datetime
    available_at: datetime
    source: SourceRef

    _utc = field_validator("occurred_at", "available_at")(utc)

    @model_validator(mode="after")
    def ordered(self) -> TimedSource:
        if self.occurred_at > self.available_at:
            raise ValueError("source cannot be available before its event")
        return self


class CoverageEvidence(ContractModel):
    """Coverage declared by an upstream verifier; never inferred from row count.

    A verification source is mandatory for full intervals, but hashing that file
    does not independently establish its semantics. Native admission remains a
    separate gate. Duplicate point records cannot supply interval coverage.
    """

    start_at: datetime
    end_at: datetime
    basis: Literal["snapshots", "full_interval"]
    scope: Literal["pnl_path", "accounting_and_interventions", "market_bars"]
    verification: TimedSource | None = None
    gap_count: int = Field(default=0, strict=True, ge=0)

    _utc = field_validator("start_at", "end_at")(utc)

    @model_validator(mode="after")
    def consistent(self) -> CoverageEvidence:
        if self.end_at <= self.start_at:
            raise ValueError("coverage interval must have positive duration")
        if self.basis == "full_interval":
            if self.gap_count or self.verification is None:
                raise ValueError("full interval requires no declared gaps and a verifier source")
            if self.verification.available_at < self.end_at:
                raise ValueError("interval verification cannot precede the covered endpoint")
        elif self.verification is not None:
            raise ValueError("snapshots cannot carry a full-interval verification")
        return self


class LabelEvidence(ContractModel):
    status: Literal["pending", "path_unobserved", "observed_hit", "verified_nonhit", "observed_endpoint"]
    value: Decimal | None = None
    observed_at: datetime | None = None
    sources: tuple[TimedSource, ...] = ()
    coverage: CoverageEvidence | None = None

    @field_validator("observed_at")
    @classmethod
    def aware(cls, value: datetime | None) -> datetime | None:
        return utc(value) if value is not None else None

    @model_validator(mode="after")
    def evidence_required(self) -> LabelEvidence:
        if self.status in {"pending", "path_unobserved"}:
            if self.value is not None or self.observed_at is not None:
                raise ValueError("unknown labels cannot carry a numeric outcome")
        elif self.value is None or self.observed_at is None or not self.sources:
            raise ValueError("observed label requires value, time and original sources")
        if self.observed_at is not None and not any(
                s.occurred_at == self.observed_at for s in self.sources):
            raise ValueError("label timestamp must match an original event")
        return self


class ForecastContract(ContractModel):
    """Versioned envelope for an individual prediction/evaluation origin.

    Target value is 0/1 for first passage, USDT change for PnL/action reward,
    and fractional log-return variance for volatility. No conversion is implicit.
    The first-passage label concerns the deployment deadline, including its end.
    """

    schema_version: Literal["neutralgrid_forecast_evidence_contract_v1"] = SCHEMA
    contract_version: str = Field(min_length=1, max_length=128)
    family: Literal["first_passage_pnl", "fixed_horizon_pnl", "volatility", "action_reward"]
    bot: BotIdentity
    account_scope: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    account_scope_basis: Literal["unverified", "credential_fingerprint", "verified_account"] = "unverified"
    measurement_basis: Literal["contract_defined", "reported_total_pnl_not_verified_equity"] = "contract_defined"
    origin_at: datetime
    target_end_at: datetime
    horizon_minutes: int | None = Field(default=None, strict=True, gt=0)
    endpoint_tolerance_seconds: int = Field(default=0, strict=True, ge=0)
    units: Literal["binary", "USDT", "fractional_log_return_variance"]
    capital_basis: Literal["invested_margin_usdt", "not_normalized", "not_applicable"]
    invested_margin_usdt: Decimal | None = Field(default=None, gt=0)
    target_pct: Decimal | None = None
    prior_target_state: Literal["unknown", "verified_not_hit", "already_hit", "not_applicable"] = "not_applicable"
    prior_target_verification: TimedSource | None = None
    feature_sources: tuple[TimedSource, ...] = Field(min_length=1)
    label: LabelEvidence
    proposed_verdict: Literal["CONTINUE", "ADJUST", "END"] | None = None
    observed_action: Literal["CONTINUE", "ADJUST", "END"] | None = None
    action_evidence: tuple[TimedSource, ...] = ()
    price_source: Literal["mark", "last", "index"] | None = None
    sampling_interval_seconds: int | None = Field(default=None, strict=True, gt=0)
    variance_aggregation: Literal["per_step", "cumulative"] | None = None
    feature_definition_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_manifest: SourceRef
    training_eligible: Literal[False] = False
    promotion_enabled: Literal[False] = False
    influence_enabled: Literal[False] = False

    _utc = field_validator("origin_at", "target_end_at")(utc)

    @model_validator(mode="after")
    def coherent(self) -> ForecastContract:
        if not self.bot.deployed_at <= self.origin_at < self.target_end_at:
            raise ValueError("origin must follow deployment and precede target endpoint")
        if any(s.occurred_at > self.origin_at or s.available_at > self.origin_at
               for s in self.feature_sources):
            raise ValueError("future or late-arriving evidence cannot enter origin features")
        for records in (self.feature_sources, self.label.sources, self.action_evidence):
            if len({s.record_id for s in records}) != len(records):
                raise ValueError("duplicate evidence identity cannot increase support")
        all_sources = self.timed_sources()
        by_id: dict[str, TimedSource] = {}
        for source in all_sources:
            previous = by_id.setdefault(source.record_id, source)
            if previous != source:
                raise ValueError("conflicting evidence identity across roles")
        if self.label.observed_at is not None and not (
                self.origin_at < self.label.observed_at <= self.target_end_at +
                timedelta(seconds=self.endpoint_tolerance_seconds)):
            raise ValueError("label lies outside its forward interval")
        if self.family == "first_passage_pnl":
            self._first_passage()
        else:
            if self.horizon_minutes is None or self.target_end_at != self.origin_at + timedelta(minutes=self.horizon_minutes):
                raise ValueError("fixed horizon must match its explicit origin and endpoint")
            if self.target_pct is not None or self.prior_target_state != "not_applicable" or self.prior_target_verification is not None:
                raise ValueError("first-passage fields cannot be reused for another target")
            if self.label.status not in {"pending", "path_unobserved", "observed_endpoint"}:
                raise ValueError("target status is incompatible with an endpoint forecast")
            if self.label.observed_at is not None and self.label.observed_at < self.target_end_at:
                raise ValueError("endpoint outcome precedes the target horizon")
            if self.family == "volatility":
                if (self.units != "fractional_log_return_variance" or self.capital_basis != "not_applicable"
                        or self.price_source is None or self.sampling_interval_seconds is None
                        or self.variance_aggregation is None):
                    raise ValueError("volatility requires explicit price, sampling, variance units and aggregation")
                if self.label.value is not None and self.label.value < 0:
                    raise ValueError("variance cannot be negative")
            elif self.units != "USDT" or self.capital_basis != "not_normalized":
                raise ValueError("PnL/action endpoint target must retain dollar-change units")
        if self.family != "volatility" and any(x is not None for x in (
                self.price_source, self.sampling_interval_seconds, self.variance_aggregation)):
            raise ValueError("volatility fields cannot silently alter another target")
        if self.family != "first_passage_pnl" and self.invested_margin_usdt is not None:
            raise ValueError("dollar and variance targets must not acquire an implicit denominator")
        if self.family != "action_reward" and (self.observed_action is not None or self.action_evidence):
            raise ValueError("only action reward can bind an observed action")
        if self.family == "action_reward":
            if bool(self.action_evidence) != (self.observed_action is not None):
                raise ValueError("observed behavior and original evidence must appear together")
            if self.label.status == "observed_endpoint":
                coverage = self.label.coverage
                if self.observed_action is None or coverage is None or coverage.scope != "accounting_and_interventions":
                    raise ValueError("action reward needs observed behavior and interval accounting")
                self._full_coverage(coverage, self.origin_at, self.label.observed_at or self.target_end_at)
        return self

    def _first_passage(self) -> None:
        if (self.target_end_at != self.bot.deployed_at + DEPLOYMENT_HORIZON
                or self.horizon_minutes is not None or self.endpoint_tolerance_seconds
                or self.target_pct != TARGET_PCT or self.units != "binary"
                or self.capital_basis != "invested_margin_usdt" or self.invested_margin_usdt is None):
            raise ValueError("FASTWIN means +3% invested margin by deployment +7h, without tolerance")
        if self.prior_target_state not in {"unknown", "verified_not_hit"}:
            raise ValueError("pre-target origins require unknown or verified no prior hit")
        verification = self.prior_target_verification
        if self.prior_target_state == "verified_not_hit":
            if verification is None or verification.available_at > self.origin_at:
                raise ValueError("prior nonhit requires verification known at origin")
        elif verification is not None:
            raise ValueError("unknown prior target state cannot carry nonhit verification")
        if self.label.status == "observed_endpoint":
            raise ValueError("endpoint PnL is not a first-passage label")
        if self.label.status == "observed_hit" and self.label.value != 1:
            raise ValueError("witnessed target hit must have label 1")
        if self.label.status == "verified_nonhit":
            if self.label.value != 0 or self.label.observed_at != self.target_end_at:
                raise ValueError("nonhit requires label 0 at the exact deadline")
            coverage = self.label.coverage
            if coverage is None or coverage.scope != "pnl_path":
                raise ValueError("nonhit needs independent full PnL-path coverage")
            self._full_coverage(coverage, self.bot.deployed_at, self.target_end_at)

    @staticmethod
    def _full_coverage(coverage: CoverageEvidence, start: datetime, end: datetime) -> None:
        if coverage.basis != "full_interval" or coverage.start_at > start or coverage.end_at < end:
            raise ValueError("sparse or partial evidence cannot prove a complete interval")

    @property
    def remaining_seconds(self) -> float:
        return (self.target_end_at - self.origin_at).total_seconds()

    @property
    def record_sha256(self) -> str:
        return fingerprint(self.model_dump(mode="json"))

    def timed_sources(self) -> tuple[TimedSource, ...]:
        values = (*self.feature_sources, *self.label.sources, *self.action_evidence)
        if self.prior_target_verification is not None:
            values += (self.prior_target_verification,)
        if self.label.coverage is not None and self.label.coverage.verification is not None:
            values += (self.label.coverage.verification,)
        return values


def verify_forecast_sources(contract: ForecastContract) -> None:
    """Check bytes only; never upgrade a producer assertion to native proof."""
    refs = [contract.dataset_manifest, *(s.source for s in contract.timed_sources())]
    seen: dict[str, str] = {}
    for ref in refs:
        path = str(Path(ref.path).resolve())
        previous = seen.setdefault(path, ref.sha256)
        if previous != ref.sha256:
            raise ValueError("one source path has conflicting hashes")
    for path, expected in seen.items():
        if file_hash(Path(path)) != expected:
            raise ValueError("forecast source hash mismatch")


def forecast_from_reported_pnl_path(
    path: Path, *, live_root: Path, account_scope: str = "unverified_reported_ui",
    origin_observation_id: str | None = None,
) -> ForecastContract:
    """Export a verified capture path into the common, permanently inert envelope.

    Default origin is the latest available pre-deadline observation before any
    witnessed hit. An explicitly selected origin must satisfy the same rule.
    Sparse history cannot establish no prior crossing, so even a witnessed future
    hit does not become a conditional-risk training label here. Every original
    drawer is reparsed; the canonical observation and originals remain referenced.
    UI source verification does not authenticate exchange account ownership.
    """
    from neutralgrid.live.decision.evidence_cohort import read_json, verify_observation_sources
    from neutralgrid.live.decision.pnl_history import validate_pnl_observation

    raw, digest = read_json(path)
    bot = BotIdentity.model_validate(raw.get("bot"))
    deadline = bot.deployed_at + DEPLOYMENT_HORIZON
    if (raw.get("schema_version") != "neutralgrid_authenticated_evidence_v1"
            or raw.get("record_type") != "timestamped_reported_pnl_path"
            or raw.get("measurement") != "reported_total_pnl_not_verified_equity"
            or raw.get("capital_basis") != "reported_invested_margin"
            or raw.get("path_coverage") != "path_unobserved_between_snapshots"
            or decimal(raw.get("target_pct")) != TARGET_PCT
            or utc(datetime.fromisoformat(str(raw.get("deadline")))) != deadline
            or raw.get("training_eligible") is not False or raw.get("influence_enabled") is not False):
        raise ValueError("reported path contract mismatch")
    if not isinstance(raw.get("points"), list) or not raw["points"]:
        raise ValueError("reported path requires original points")
    unique: dict[str, tuple[TimedSource, Decimal, Decimal, tuple[SourceRef, ...]]] = {}
    for point in raw["points"]:
        if not isinstance(point, dict):
            raise ValueError("malformed reported point")
        timed = TimedSource(record_id=point["observation_id"], occurred_at=point["occurred_at"],
                            available_at=point["available_at"], source=point["observation_source"])
        original, checksum = read_json(Path(timed.source.path))
        if checksum != timed.source.sha256:
            raise ValueError("reported point observation changed")
        record = validate_pnl_observation(original)
        actual = BotIdentity(symbol=record["symbol"], strategy_id=record["strategy_id"],
                             deployed_at=record["deploy_ts"])
        pnl = decimal(point.get("reported_total_pnl_usdt"))
        margin = decimal(point.get("reported_invested_margin_usdt"))
        if (actual != bot or timed.record_id != record["observation_id"]
                or timed.occurred_at != record["captured_at"]
                or timed.available_at != record["evaluated_at"]
                or pnl != decimal(record["pnl"].get("total_profit_usdt"))
                or margin != decimal(record["grid"].get("invested_margin_usdt")) or margin <= 0):
            raise ValueError("reported point does not match original identity, timing or accounting")
        native = tuple(SourceRef.model_validate(ref) for ref in
                       verify_observation_sources(record, live_root=live_root))
        declared = tuple(SourceRef.model_validate(ref) for ref in point.get("original_sources", []))
        if declared != native:
            raise ValueError("reported point original-source chain mismatch")
        item = (timed, pnl, margin, native)
        prior = unique.setdefault(timed.record_id, item)
        if prior != item:
            raise ValueError("conflicting reported point identity")
    margins = {item[2] for item in unique.values()}
    if len(margins) != 1:
        raise ValueError("reported capital basis changed; target denominator unresolved")
    margin = next(iter(margins))
    ordered = sorted(unique.values(), key=lambda item: (item[0].available_at, item[0].record_id))
    hits = [item[0].occurred_at for item in ordered if item[1] * 100 >= TARGET_PCT * margin
            and item[0].occurred_at <= deadline]
    first_witnessed_hit = min(hits) if hits else None
    eligible = [item for item in ordered if item[0].available_at < deadline
                and (first_witnessed_hit is None or item[0].available_at < first_witnessed_hit)]
    if origin_observation_id is not None:
        eligible = [item for item in eligible if item[0].record_id == origin_observation_id]
    if not eligible:
        raise ValueError("no pre-deadline origin before the first witnessed hit")
    origin = eligible[-1][0].available_at
    sources: list[TimedSource] = []
    for timed, _, _, refs in ordered:
        if timed.available_at <= origin:
            sources.append(timed)
            sources.extend(TimedSource(record_id=f"{timed.record_id}:original:{index}",
                                       occurred_at=timed.occurred_at, available_at=timed.available_at,
                                       source=ref) for index, ref in enumerate(refs))
    envelope = ForecastContract(
        contract_version="reported_pnl_pre_target_3pct_by_deployment_7h_v1",
        family="first_passage_pnl", bot=bot, account_scope=account_scope,
        account_scope_basis="unverified", measurement_basis="reported_total_pnl_not_verified_equity",
        origin_at=origin, target_end_at=deadline, units="binary",
        capital_basis="invested_margin_usdt", invested_margin_usdt=margin, target_pct=TARGET_PCT,
        prior_target_state="unknown", feature_sources=tuple(sources),
        label=LabelEvidence(status="path_unobserved"),
        feature_definition_sha256=fingerprint({"version": "reported_pnl_path_features_v1",
            "fields": ["reported_total_pnl_usdt", "reported_invested_margin_usdt"],
            "time_rule": "event_and_availability_at_or_before_origin", "native_equity": False}),
        dataset_manifest=SourceRef(path=str(path.resolve()), sha256=digest),
    )
    verify_forecast_sources(envelope)
    return envelope
