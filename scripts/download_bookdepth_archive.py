"""Download Binance Vision futures bookDepth files for candidate windows."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutralgrid.data.bookdepth_archive import (  # noqa: E402
    BOOKDEPTH_SCHEMA_VERSION,
    BookDepthFileRequest,
    file_requests_for_targets,
    load_bookdepth_targets,
    local_bookdepth_zip,
    parse_checksum_text,
    sha256_file,
    verify_checksum,
)


def _git_output(args: list[str]) -> str | None:
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False, capture_output=True, text=True)
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _download(url: str, destination: Path, *, timeout_seconds: float, retries: int) -> tuple[bool, str | None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
                with temp_path.open("wb") as handle:
                    shutil.copyfileobj(response, handle)
            temp_path.replace(destination)
            return True, None
        except urllib.error.HTTPError as exc:
            if temp_path.exists():
                temp_path.unlink()
            return False, f"http_{exc.code}"
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink()
            if attempt >= retries:
                return False, repr(exc)
            time.sleep(min(2.0 * attempt, 10.0))
    return False, "retry_exhausted"


def _handle_request(
    request: BookDepthFileRequest,
    *,
    archive_root: Path,
    timeout_seconds: float,
    retries: int,
    skip_existing: bool,
) -> dict[str, Any]:
    zip_path = local_bookdepth_zip(archive_root, request.symbol, request.date)
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".CHECKSUM")
    result: dict[str, Any] = {
        "symbol": request.symbol,
        "date": request.date.isoformat(),
        "zip_path": str(zip_path),
        "url": request.url,
        "checksum_url": request.checksum_url,
        "zip_exists_before": zip_path.exists(),
        "checksum_exists_before": checksum_path.exists(),
        "status": None,
        "checksum_match": None,
        "expected_sha256": None,
        "actual_sha256": None,
        "error": None,
    }

    if skip_existing and zip_path.exists() and checksum_path.exists():
        ok, expected, actual = verify_checksum(zip_path, checksum_path)
        result.update(
            {
                "status": "existing_verified" if ok else "existing_checksum_mismatch",
                "checksum_match": ok,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "zip_bytes": zip_path.stat().st_size if zip_path.exists() else None,
            }
        )
        if ok:
            return result

    checksum_ok, checksum_error = _download(
        request.checksum_url,
        checksum_path,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    if not checksum_ok:
        result.update({"status": "checksum_download_failed", "error": checksum_error})
        return result

    expected = parse_checksum_text(checksum_path.read_text(encoding="utf-8", errors="replace"))
    zip_ok, zip_error = _download(request.url, zip_path, timeout_seconds=timeout_seconds, retries=retries)
    if not zip_ok:
        result.update({"status": "zip_download_failed", "error": zip_error, "expected_sha256": expected})
        return result

    actual = sha256_file(zip_path)
    match = expected is not None and actual.lower() == expected.lower()
    result.update(
        {
            "status": "downloaded_verified" if match else "downloaded_checksum_mismatch",
            "checksum_match": match,
            "expected_sha256": expected,
            "actual_sha256": actual,
            "zip_bytes": zip_path.stat().st_size,
        }
    )
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Candidate CSV/XLSX/Parquet with symbol and scan timestamp")
    parser.add_argument("--archive-root", default=str(ROOT / "data" / "book_depth_archive" / "fastwin_pool"))
    parser.add_argument("--output", default=None, help="Download manifest path")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--fallback-position-usdt", type=float, default=None)
    parser.add_argument("--lookback-hours", type=float, default=1.0)
    parser.add_argument("--forward-hours", type=float, default=7.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    archive_root = Path(args.archive_root)
    targets = load_bookdepth_targets(
        input_path,
        max_candidates=args.max_candidates,
        fallback_position_usdt=args.fallback_position_usdt,
    )
    requests = file_requests_for_targets(
        targets,
        lookback_hours=args.lookback_hours,
        forward_hours=args.forward_hours,
    )
    output = Path(args.output) if args.output else archive_root / "download_manifest.json"

    results: list[dict[str, Any]] = []
    if args.dry_run:
        for request in requests:
            zip_path = local_bookdepth_zip(archive_root, request.symbol, request.date)
            results.append(
                {
                    "symbol": request.symbol,
                    "date": request.date.isoformat(),
                    "zip_path": str(zip_path),
                    "url": request.url,
                    "checksum_url": request.checksum_url,
                    "zip_exists_before": zip_path.exists(),
                    "status": "dry_run",
                }
            )
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as executor:
            futures = [
                executor.submit(
                    _handle_request,
                    request,
                    archive_root=archive_root,
                    timeout_seconds=float(args.timeout_seconds),
                    retries=max(1, int(args.retries)),
                    skip_existing=not args.no_skip_existing,
                )
                for request in requests
            ]
            for future in as_completed(futures):
                results.append(future.result())

    status_counts: dict[str, int] = {}
    for result in results:
        status = str(result.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    manifest = {
        "schema_version": BOOKDEPTH_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "input_path": str(input_path),
        "archive_root": str(archive_root),
        "target_count": len(targets),
        "request_count": len(requests),
        "lookback_hours": float(args.lookback_hours),
        "forward_hours": float(args.forward_hours),
        "dry_run": bool(args.dry_run),
        "status_counts": status_counts,
        "git_head": _git_output(["rev-parse", "--short", "HEAD"]),
        "git_status_short": _git_output(["status", "--short"]),
        "results": sorted(results, key=lambda row: (str(row.get("symbol")), str(row.get("date")))),
    }
    _write_json(output, manifest)

    failures = [
        row
        for row in results
        if str(row.get("status")) not in {"dry_run", "existing_verified", "downloaded_verified"}
    ]
    print(
        json.dumps(
            {
                "manifest": str(output),
                "target_count": len(targets),
                "request_count": len(requests),
                "status_counts": status_counts,
                "failure_count": len(failures),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
