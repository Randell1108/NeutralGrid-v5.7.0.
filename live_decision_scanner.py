"""Tactical live decision scanner -- CLI entry point.

Modes
-----
* ``--dry-run`` (Phase A): load YAML, print synthetic recommendations, exit.
* ``--once`` (Phase B): one live tick -- fetch Binance, evaluate, render to
  console + append to ``logs/live_decisions_YYYYMMDD.jsonl``.
* recurring loop (Phase C, default): one tick every ``--interval``; ctrl-C
  drains the in-flight tick, persists state, and exits cleanly. A Discord
  digest is posted per tick when ``NEUTRALGRID_DISCORD_WEBHOOK`` is set
  (or ``--discord-webhook`` is passed).

See LIVE_DECISION.md for the full design.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Force .env load so NEUTRALGRID_DISCORD_WEBHOOK is visible. The project's
# config module triggers this on import, but doing it explicitly here keeps
# the dependency chain clear.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover -- python-dotenv is in requirements
    pass

from neutralgrid.api.binance_client import BinanceClient
from neutralgrid.core.constants import DECISION_CONTRACT_VERSION
from neutralgrid.live.decision.discord_sink import DiscordDigestSink
from neutralgrid.live.decision.loader import (
    LiveBotSpec,
    LoaderError,
    find_active_yaml_paths,
    is_dated_registry_yaml,
    load_bot_specs,
)
from neutralgrid.live.decision.monitor import MonitorContext, evaluate_bot
from neutralgrid.live.decision.recommender import (
    BotEvaluation,
    RecommenderConfig,
    decide,
)
from neutralgrid.live.decision.renderer import (
    ConsoleRenderer,
    JsonlWriter,
    ScanResult,
)
from neutralgrid.live.decision.state_store import (
    BotHistory,
    DEFAULT_STATE_DIR,
    load_history,
    save_history,
)

logger = logging.getLogger("live_decision_scanner")

DEFAULT_BOTS_DIR = Path("active bots")
DEFAULT_INTERVAL = "5m"
DISCORD_WEBHOOK_ENV_VAR = "NEUTRALGRID_DISCORD_WEBHOOK"
MIN_INTERVAL_SECONDS = 30  # 30s floor avoids accidental rate-limit storms


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="live_decision_scanner",
        description=(
            "Recurring CONTINUE / ADJUST / END recommender for live grid bots. "
            "Modes: --dry-run, --once, or default recurring loop. "
            "See LIVE_DECISION.md."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Live-bot YAMLs: one active symbol per file, e.g. RENDERUSDT.yaml. "
            "Legacy DD-MM-YY.yaml files are still supported when no symbol files exist.\n"
            f"Decision contract version: {DECISION_CONTRACT_VERSION}\n"
            f"Discord webhook: --discord-webhook or env var ${DISCORD_WEBHOOK_ENV_VAR}\n"
        ),
    )
    p.add_argument(
        "--bots",
        type=Path,
        default=None,
        help="Path to a specific YAML file (overrides --bots-dir).",
    )
    p.add_argument(
        "--bots-dir",
        type=Path,
        default=DEFAULT_BOTS_DIR,
        help=(
            "Directory containing active per-symbol YAML files. Falls back to "
            f"legacy DD-MM-YY.yaml files when no symbol files exist (default: {DEFAULT_BOTS_DIR})."
        ),
    )
    p.add_argument(
        "--state-dir",
        type=Path,
        default=DEFAULT_STATE_DIR,
        help=f"Per-bot history directory (default: {DEFAULT_STATE_DIR}).",
    )
    p.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs"),
        help="Where the JSONL audit log is appended (default: logs/).",
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Phase A: load YAML, print synthetic recommendations, exit.",
    )
    mode.add_argument(
        "--once",
        action="store_true",
        help="Phase B: one live tick -- fetch, evaluate, render, append, exit.",
    )

    p.add_argument(
        "--interval",
        type=str,
        default=DEFAULT_INTERVAL,
        help=(
            "Recurring tick cadence in the loop mode. Accepts '5m', '300s', "
            f"'1h', or a bare number (seconds). Default: {DEFAULT_INTERVAL}. "
            f"Floor: {MIN_INTERVAL_SECONDS}s."
        ),
    )
    p.add_argument(
        "--discord-webhook",
        type=str,
        default=None,
        help=(
            "Discord webhook URL. Falls back to "
            f"${DISCORD_WEBHOOK_ENV_VAR} env var. Missing webhook = warn-and-"
            "continue (console + JSONL still flow)."
        ),
    )
    p.add_argument(
        "--no-discord",
        action="store_true",
        help="Suppress Discord output even if a webhook is configured.",
    )
    p.add_argument(
        "--meta-labeler-path",
        type=Path,
        default=None,
        help="Override path to the meta-labeler artifact (default: models/meta_labeler.pkl).",
    )
    p.add_argument(
        "--config-file",
        type=Path,
        default=None,
        help=(
            "YAML/JSON file overriding RecommenderConfig defaults "
            "(e.g. min_range_prob: 0.50). Unknown keys are ignored with a warning."
        ),
    )
    p.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return p


# -- Argument parsing helpers ------------------------------------------------


def parse_interval(s: str) -> float:
    """Parse '5m', '300s', '1h', or bare seconds. Returns seconds as float.

    Raises ValueError on malformed or non-positive intervals.
    """
    s = s.strip().lower()
    if not s:
        raise ValueError("--interval cannot be empty")
    multiplier = 1.0
    body = s
    for suffix, mult in (("ms", 0.001), ("s", 1.0), ("m", 60.0), ("h", 3600.0)):
        if s.endswith(suffix):
            body = s[: -len(suffix)]
            multiplier = mult
            break
    try:
        value = float(body)
    except ValueError as e:
        raise ValueError(f"--interval not a number: {s!r}") from e
    seconds = value * multiplier
    if seconds <= 0:
        raise ValueError(f"--interval must be positive, got {seconds}s")
    return seconds


def _resolve_yaml_paths(args: argparse.Namespace) -> list[Path]:
    if args.bots is not None:
        return [Path(args.bots)]
    return find_active_yaml_paths(Path(args.bots_dir))


def _resolve_webhook(args: argparse.Namespace) -> Optional[str]:
    if args.no_discord:
        return None
    if args.discord_webhook:
        return args.discord_webhook
    return os.environ.get(DISCORD_WEBHOOK_ENV_VAR) or None


def _synthetic_evaluation(spec: LiveBotSpec, now: datetime) -> BotEvaluation:
    midpoint = (spec.grid_lower + spec.grid_upper) / 2.0
    return BotEvaluation(
        symbol=spec.symbol,
        evaluated_at_utc=now,
        grid_lower=spec.grid_lower,
        grid_upper=spec.grid_upper,
        deploy_ts=spec.deploy_ts,
        price=midpoint,
        pct_inside_grid=50.0,
        dist_to_lower_pct=50.0,
        dist_to_upper_pct=50.0,
        range_prob=0.65,
        trend_prob=0.20,
        persistence_prob=0.55,
        meta_proba=0.55,
        utility_score=0.10,
        micro_gate_pass=True,
    )


# -- Lock file ---------------------------------------------------------------


class LockFile:
    """File-based mutex preventing two scanner instances on the same state dir.

    Cross-platform: stores ``pid\\nstart_time_utc`` in the lock file. We do
    NOT auto-detect stale locks; on conflict we instruct the user to remove
    the file manually if they're sure no scanner is running. Auto-detection
    requires probing PID liveness which is OS-dependent and risks false
    positives on PID reuse.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._held = False

    def acquire(self) -> bool:
        # O_CREAT | O_EXCL is the canonical race-free "create only if absent"
        # primitive on both POSIX and Windows; it converts the file-exists
        # check into a single atomic syscall (closes the TOCTOU window the
        # earlier exists()/write_text() pattern had).
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"{os.getpid()}\n{datetime.now(timezone.utc).isoformat()}\n"
        try:
            fd = os.open(
                str(self.path),
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o644,
            )
        except FileExistsError:
            try:
                content = self.path.read_text(encoding="utf-8").strip()
            except Exception:
                content = "<unreadable>"
            logger.error(
                "Scanner lock file at %s already exists:\n  %s\n"
                "Another scanner may be running. If you are sure none is, "
                "delete the file manually and re-run.",
                self.path, content,
            )
            return False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
        except Exception:
            # Best-effort cleanup so a half-written lock doesn't strand subsequent runs.
            with suppress(OSError, FileNotFoundError):
                self.path.unlink()
            raise
        self._held = True
        return True

    def release(self) -> None:
        if not self._held:
            return
        with suppress(FileNotFoundError, OSError):
            self.path.unlink()
        self._held = False


# -- Tick / loop --------------------------------------------------------------


async def _run_tick(
    args: argparse.Namespace,
    yaml_paths: list[Path],
    *,
    context: MonitorContext,
    client: BinanceClient,
    cfg: RecommenderConfig,
    now: datetime,
) -> list[ScanResult]:
    """Run one full live tick. Loads the YAML fresh each call so user edits
    take effect on the next tick.
    """
    specs, issues = _load_specs_from_yaml_paths(
        yaml_paths,
        now=now,
        enforce_symbol_filenames=args.bots is None,
    )
    for issue in issues:
        logger.warning("%s", issue)

    results: list[ScanResult] = []
    for spec in specs:
        history = load_history(spec.state_key, base_dir=args.state_dir)
        evaluation = await evaluate_bot(spec, context=context, client=client, now=now)
        decision = decide(evaluation, history, cfg, now)
        save_history(spec.state_key, decision.new_history, base_dir=args.state_dir)
        results.append(
            ScanResult(
                spec=spec,
                evaluation=evaluation,
                recommendation=decision.recommendation,
                should_emit=decision.should_emit,
            )
        )
    return results


def _load_specs_from_yaml_paths(
    yaml_paths: list[Path],
    *,
    now: datetime,
    enforce_symbol_filenames: bool,
) -> tuple[list[LiveBotSpec], list[str]]:
    specs: list[LiveBotSpec] = []
    issues: list[str] = []

    for yaml_path in yaml_paths:
        try:
            loaded, warnings = load_bot_specs(yaml_path, now=now)
        except OSError as e:
            issues.append(f"[loader yaml_read_failed] {yaml_path}: {e}")
            continue
        except LoaderError as e:
            issues.append(f"[loader yaml_parse_failed] {yaml_path}: {e}")
            continue

        for w in warnings:
            idx = f"#{w.bot_index}" if w.bot_index is not None else ""
            issues.append(f"[loader {w.code}] {yaml_path} {idx} {w.message}")

        if enforce_symbol_filenames and not is_dated_registry_yaml(yaml_path):
            if len(loaded) != 1:
                issues.append(
                    f"[loader active_symbol_file_error] {yaml_path} loaded "
                    f"{len(loaded)} bots; expected exactly one bot"
                )
                continue
            expected_symbol = yaml_path.stem.upper()
            actual_symbol = loaded[0].symbol.upper()
            if actual_symbol != expected_symbol:
                issues.append(
                    f"[loader active_symbol_file_error] {yaml_path} filename "
                    f"symbol {expected_symbol} does not match YAML symbol {actual_symbol}"
                )
                continue

        specs.extend(loaded)

    return specs, issues


def _load_recommender_config(args: argparse.Namespace) -> RecommenderConfig:
    """Resolve `RecommenderConfig` from --config-file or fall back to defaults.

    Fail-closed: an explicitly passed --config-file that cannot be loaded is
    an operator error, not a fallback case — silently running on default
    thresholds (emit cadence, meta-tilt settings) would misconfigure every
    subsequent tick, unnoticed in scheduled/loop mode.
    """
    if args.config_file is None:
        return RecommenderConfig()
    try:
        cfg = RecommenderConfig.from_file(args.config_file)
        logger.info("Loaded RecommenderConfig overrides from %s", args.config_file)
        return cfg
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(
            f"ERROR: failed to load --config-file {args.config_file}: {e}"
        ) from e


async def _run_once(args: argparse.Namespace, yaml_paths: list[Path]) -> int:
    cfg = _load_recommender_config(args)
    context = MonitorContext.create(meta_labeler_path=args.meta_labeler_path)
    client = BinanceClient()
    sink = DiscordDigestSink(_resolve_webhook(args))
    renderer = ConsoleRenderer()

    now = datetime.now(timezone.utc)
    try:
        results = await _run_tick(
            args, yaml_paths, context=context, client=client, cfg=cfg, now=now
        )
        print(renderer.render(results, now=now))
        with JsonlWriter(base_dir=args.logs_dir) as jsonl:
            for r in results:
                jsonl.append(r, now=now)
        if results:
            await sink.send(results, now=now)
    finally:
        await sink.aclose()
        with suppress(Exception):
            await client.close()
    return 0


async def _run_loop(args: argparse.Namespace) -> int:
    """Recurring tick loop with graceful shutdown on SIGINT/SIGTERM."""
    try:
        interval_s = parse_interval(args.interval)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if interval_s < MIN_INTERVAL_SECONDS:
        print(
            f"ERROR: --interval must be >= {MIN_INTERVAL_SECONDS}s "
            f"(got {interval_s}s). Lower cadences risk Binance rate-limit storms.",
            file=sys.stderr,
        )
        return 2

    # Config loads BEFORE the lock: a fail-closed SystemExit on a bad
    # --config-file must not strand .scanner.lock (config loading is
    # side-effect-free and order-independent).
    cfg = _load_recommender_config(args)

    lock = LockFile(args.state_dir.parent / ".scanner.lock")
    if not lock.acquire():
        return 1

    context = MonitorContext.create(meta_labeler_path=args.meta_labeler_path)
    client = BinanceClient()
    sink = DiscordDigestSink(_resolve_webhook(args))
    renderer = ConsoleRenderer()

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    force_exit_count = [0]  # mutable closure cell

    def _request_stop_async(signame: str) -> None:
        """Runs in the asyncio loop thread (loop.add_signal_handler path)."""
        if stop_event.is_set():
            force_exit_count[0] += 1
            if force_exit_count[0] >= 1:
                logger.warning(
                    "Second %s received; forcing immediate exit. "
                    "Lock file may need manual cleanup.",
                    signame,
                )
                # Cancel all running tasks; the finally block in _run_loop
                # will run as the task unwinds, releasing lock + closing
                # client. This is preferable to sys.exit() which on Windows
                # would skip the finally block when raised mid-task.
                for task in asyncio.all_tasks(loop):
                    task.cancel()
            return
        logger.info("%s received; draining in-flight tick before exit.", signame)
        stop_event.set()

    def _request_stop_threadsafe(signame: str) -> None:
        """Runs in the OS main thread (signal.signal fallback on Windows).

        Marshals back into the asyncio loop so stop_event manipulation and
        task cancellation are safe.
        """
        loop.call_soon_threadsafe(_request_stop_async, signame)

    # Windows: signal.SIGINT works; SIGTERM is not delivered the same way but
    # we install the handler unconditionally so POSIX gets graceful shutdown.
    for signame in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, signame, None)
        if sig is None:
            continue
        try:
            loop.add_signal_handler(sig, _request_stop_async, signame)
        except (NotImplementedError, RuntimeError):
            # Windows asyncio loop doesn't support add_signal_handler; fall
            # back to signal.signal + call_soon_threadsafe so the handler
            # runs in the loop thread and the finally block executes cleanly.
            with suppress(ValueError, OSError):
                signal.signal(
                    sig,
                    lambda _s, _f, name=signame: _request_stop_threadsafe(name),
                )

    logger.info(
        "Loop start: interval=%.0fs, contract=%s, webhook=%s",
        interval_s,
        DECISION_CONTRACT_VERSION,
        "set" if _resolve_webhook(args) else "unset",
    )

    try:
        with JsonlWriter(base_dir=args.logs_dir) as jsonl:
            while not stop_event.is_set():
                tick_start = datetime.now(timezone.utc)

                yaml_paths = _resolve_yaml_paths(args)
                if not yaml_paths or not all(path.is_file() for path in yaml_paths):
                    logger.warning(
                        "no_active_bots: no YAML file found at %s",
                        args.bots or args.bots_dir,
                    )
                    results: list[ScanResult] = []
                else:
                    results = await _run_tick(
                        args, yaml_paths,
                        context=context, client=client, cfg=cfg, now=tick_start,
                    )

                print(renderer.render(results, now=tick_start))
                if results:
                    for r in results:
                        jsonl.append(r, now=tick_start)
                    with suppress(Exception):
                        await sink.send(results, now=tick_start)

                # Wait for next tick or shutdown signal
                with suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
    finally:
        logger.info("Loop draining; closing sinks.")
        await sink.aclose()
        with suppress(Exception):
            await client.close()
        lock.release()

    logger.info("Loop exit clean.")
    return 0


# -- Dry-run (Phase A) --------------------------------------------------------


def _run_dry_run(yaml_paths: list[Path], *, enforce_symbol_filenames: bool) -> int:
    now = datetime.now(timezone.utc)
    specs, issues = _load_specs_from_yaml_paths(
        yaml_paths,
        now=now,
        enforce_symbol_filenames=enforce_symbol_filenames,
    )

    print("== live_decision_scanner --dry-run ==")
    print(f"contract: {DECISION_CONTRACT_VERSION}")
    if len(yaml_paths) == 1:
        print(f"yaml:     {yaml_paths[0]}")
    else:
        print("yamls:")
        for yaml_path in yaml_paths:
            print(f"  - {yaml_path}")
    print(f"now:      {now.isoformat()}")
    print(f"bots:     {len(specs)} loaded, {len(issues)} warning(s)")

    for issue in issues:
        print(f"  {issue}")

    if not specs:
        print("(no bots to evaluate)")
        return 0

    cfg = RecommenderConfig()
    print()
    print(f"{'SYMBOL':<12} {'STATE_KEY':<32} {'VERDICT':<10} REASONS")
    print("-" * 72)
    for spec in specs:
        evaluation = _synthetic_evaluation(spec, now)
        result = decide(evaluation, BotHistory.empty(), cfg, now)
        rec = result.recommendation
        reasons = ", ".join(rec.reasons) if rec.reasons else "-"
        key_disp = spec.state_key if len(spec.state_key) <= 30 else spec.state_key[:27] + "..."
        print(f"{spec.symbol:<12} {key_disp:<32} {rec.verdict.value:<10} {reasons}")

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    # Avoid leaking the Discord webhook URL (or any other URL) into logs.
    # httpx logs every request at INFO; keep it at WARNING unless the user
    # explicitly asks for DEBUG.
    if args.log_level != "DEBUG":
        logging.getLogger("httpx").setLevel(logging.WARNING)

    yaml_paths = _resolve_yaml_paths(args)

    if args.dry_run:
        if not yaml_paths or not all(path.is_file() for path in yaml_paths):
            _print_no_yaml(args)
            return 1
        return _run_dry_run(yaml_paths, enforce_symbol_filenames=args.bots is None)

    if args.once:
        if not yaml_paths or not all(path.is_file() for path in yaml_paths):
            _print_no_yaml(args)
            return 1
        return asyncio.run(_run_once(args, yaml_paths))

    # Default: recurring loop. We don't refuse-to-start on missing YAML --
    # the loop tolerates an empty 'active bots' dir and waits for one to
    # appear (matching the "user edits YAML mid-tick" UX).
    return asyncio.run(_run_loop(args))


def _print_no_yaml(args: argparse.Namespace) -> None:
    print(
        f"No YAML file found (searched: {args.bots or args.bots_dir}).\n"
        f"Create one active symbol file, e.g. '{args.bots_dir}/RENDERUSDT.yaml', "
        "or use a legacy '<DD-MM-YY>.yaml' file with a top-level 'bots:' list. "
        "See LIVE_DECISION.md for the schema.",
        file=sys.stderr,
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
