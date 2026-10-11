"""Bounded past-only path ablation; signed signatures are not return evidence."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PricePoint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    captured_at: datetime
    available_at: datetime
    price: float = Field(gt=0)
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("captured_at", "available_at")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("price timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @field_validator("price", mode="before")
    @classmethod
    def numeric(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("boolean is not a price")
        return value


def signature_level_two(points: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if len(points) < 2 or len(points[0]) != 2:
        raise ValueError("signature requires a two-channel path")
    first = [0.0, 0.0]
    second = [[0.0, 0.0], [0.0, 0.0]]
    for previous, current in zip(points, points[1:]):
        if len(previous) != 2 or len(current) != 2 or not all(math.isfinite(v) for v in (*previous, *current)):
            raise ValueError("invalid path point")
        delta = [current[i] - previous[i] for i in range(2)]
        for i in range(2):
            for j in range(2):
                second[i][j] += first[i] * delta[j] + 0.5 * delta[i] * delta[j]
        first = [first[i] + delta[i] for i in range(2)]
    result = tuple(first + [v for row in second for v in row])
    if not all(math.isfinite(v) for v in result):
        raise ValueError("signature overflow")
    return result


SUMMARY_COLUMNS = ("path_return", "path_realized_vol", "path_min", "path_max", "path_variation")
SIGNATURE_COLUMNS = ("sig_time", "sig_price", "sig_time_time", "sig_time_price", "sig_price_time", "sig_price_price")


def path_features(points: Sequence[PricePoint], *, asof: datetime) -> dict[str, float]:
    if asof.tzinfo is None:
        raise ValueError("asof must include timezone")
    start = asof - timedelta(minutes=60)
    # Future records are ignored, including future values perturbed in leakage tests.
    past = sorted((p for p in points if start <= p.captured_at <= asof and
                   p.captured_at <= p.available_at <= asof), key=lambda p: p.captured_at)
    if len(past) < 13 or past[0].captured_at != start or past[-1].captured_at != asof:
        raise ValueError("full observed 60-minute path endpoints required")
    if any(not timedelta(0) < b.captured_at - a.captured_at <= timedelta(minutes=5)
           for a, b in zip(past, past[1:])):
        raise ValueError("duplicate timestamp or path gap exceeds five minutes")
    coords = [((p.captured_at - start).total_seconds() / 3600.0,
               math.log(p.price) - math.log(past[0].price)) for p in past]
    changes = [b[1] - a[1] for a, b in zip(coords, coords[1:])]
    values = [coords[-1][1], math.sqrt(sum(v*v for v in changes)),
              min(v[1] for v in coords), max(v[1] for v in coords), sum(abs(v) for v in changes)]
    return dict(zip(SUMMARY_COLUMNS + SIGNATURE_COLUMNS, values + list(signature_level_two(coords))))
