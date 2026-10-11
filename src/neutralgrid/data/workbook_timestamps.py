"""Canonical Excel end timestamps for expired-bot workbook writers."""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from numbers import Real
from typing import Any

from openpyxl.utils.datetime import CALENDAR_WINDOWS_1900, from_excel

END_TIME_NUMBER_FORMAT = "yyyy-mm-dd hh:mm:ss"


def parse_workbook_utc(value: Any, epoch: datetime = CALENDAR_WINDOWS_1900) -> datetime | None:
    """Accept Excel serials and explicit year-first timestamps; return UTC naive."""
    if isinstance(value, Real) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return None
        try:
            value = from_excel(float(value), epoch)
        except (ValueError, OverflowError):
            return None
    if isinstance(value, str):
        value = value.strip()
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            for pattern in ("%Y/%m/%d %H/%M/%S", "%Y/%m/%d %H:%M:%S"):
                try:
                    value = datetime.strptime(value, pattern)
                    break
                except ValueError:
                    continue
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def normalize_end_time_utc(
    end: Any, start: Any, duration_hours: Any,
    epoch: datetime = CALENDAR_WINDOWS_1900,
) -> datetime | None:
    """Preserve a valid recorded end; otherwise derive only from known inputs."""
    parsed_start = parse_workbook_utc(start, epoch)
    parsed_end = parse_workbook_utc(end, epoch)
    if parsed_end is not None and (parsed_start is None or parsed_end >= parsed_start):
        return parsed_end
    try:
        hours = float(duration_hours)
        if parsed_start is not None and math.isfinite(hours) and hours >= 0:
            return parsed_start + timedelta(hours=hours)
    except (TypeError, ValueError, OverflowError):
        pass
    return None
