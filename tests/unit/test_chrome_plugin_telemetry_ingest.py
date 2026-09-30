from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from scripts import ingest_chrome_plugin_telemetry_cycle as ingest
from scripts import run_live_telemetry_controller as controller
from scripts import run_drawer_verdict_cycle as recurring


UTC = timezone.utc


@pytest.mark.parametrize("missing_l2", [False, True])
def test_ingest_to_recurring_controller_commits_once_without_action_dispatch(tmp_path, monkeypatch, missing_l2):
    bundle, _, _ = _write_bundle(tmp_path)
    live = tmp_path / "Live"
    manifest = ingest.ingest_capture_bundle(bundle, workspace_root=tmp_path,
        live_root=live, audit_dir=tmp_path / "outputs/audits/plugin_cycles")
    now = datetime(2026, 8, 13, 15, 0, 35, tzinfo=UTC)
    monkeypatch.setattr(recurring, "_utc_now", lambda: now)
    monkeypatch.setattr(controller, "_utc_now", lambda: now)
    run_dir = tmp_path / "public_run"
    run_dir.mkdir()
    (run_dir / "l2_risk_snapshots.jsonl").write_text("{}\n", encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({"symbol": "BTCUSDT", "run_id": "run-1"}), encoding="utf-8")
    public_manifest = tmp_path / "public.json"
    public_manifest.write_text(json.dumps({"run_id": "run-1", "symbol_run_dirs": {"BTCUSDT": str(run_dir)}}), encoding="utf-8")
    cycle = controller.load_complete_cycle(manifest)
    calls = []
    def scanner(command, **kwargs):
        calls.append(command)
        registry_path = Path(command[command.index("--bots") + 1])
        assert registry_path.is_file()
        row = {"ts": now.isoformat(), "symbol": "BTCUSDT", "strategy_id": "413500001",
                 "verdict": "END", "reasons": ["synthetic_test_only"],
                 "execution_telemetry": cycle.bots[0].scanner_entry["execution_telemetry"],
                 "evaluation": {"evaluated_at_utc": now.isoformat(), "price": 1.5, "diagnostics": [],
                                "l2_risk": None if missing_l2 else {"run_id": "run-1", "age_seconds": 0.0,
                                                                    "captured_at_utc": now.isoformat()}}}
        log_dir = Path(command[command.index("--logs-dir") + 1])
        (log_dir / "live_decisions_20260813.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(controller, "_run_process", scanner)
    def forbidden(*a, **kw):
        pytest.fail("advisory consumer dispatched an action")
    monkeypatch.setattr(controller, "route_actions", forbidden)
    args = recurring.parse_args(["--cycle-manifest", str(manifest), "--live-root", str(live),
                                 "--audit-dir", str(tmp_path / "consumer"),
                                 "--diff-depth-manifest", str(public_manifest)])
    result = recurring.consume(args)
    if missing_l2:
        assert result["status"] == "blocked"
        assert result["verdicts"] == []
        assert recurring.health(args.audit_dir, now=now)["verdicts"] == []
        assert recurring.consume(args)["status"] == "blocked"
        assert len(calls) == 1
        assert not list(live.rglob("pnl_history/*/observations/*.json"))
        return
    assert result["status"] == "complete", result
    assert result["controller_report"]["pnl_history_appended"] == 1
    assert result["controller_report"]["action_outcomes"][0]["status"] == "observational_not_executed"
    assert recurring.consume(args)["status"] == "duplicate"
    assert len(calls) == 1
    assert len(list(live.rglob("pnl_history/*/observations/*.json"))) == 1


def test_recurring_raw_capture_is_read_from_exact_live_run_directory(tmp_path: Path) -> None:
    bundle, raw, payload = _write_bundle(tmp_path)
    live_raw = tmp_path / "Live/2026-08-13/BTCUSDT/drawer_captures/run-1/drawer.txt"
    live_raw.parent.mkdir(parents=True)
    live_raw.write_bytes(raw.read_bytes())
    payload["captures"][0]["raw_text_path"] = str(live_raw)
    bundle.write_text(json.dumps(payload), encoding="utf-8")
    path = ingest.ingest_capture_bundle(bundle, workspace_root=tmp_path,
        live_root=tmp_path / "Live", audit_dir=tmp_path / "outputs/audits/plugin_cycles")
    assert controller.load_complete_cycle(path).symbols == ("BTCUSDT",)


@pytest.mark.parametrize("suffix", [
    "2026-08-12/BTCUSDT/drawer_captures/run-1",
    "2026-08-13/ETHUSDT/drawer_captures/run-1",
    "2026-08-13/BTCUSDT/drawer_captures/another-run",
])
def test_recurring_raw_capture_rejects_wrong_date_symbol_or_run(tmp_path: Path, suffix: str) -> None:
    bundle, raw, payload = _write_bundle(tmp_path)
    wrong = tmp_path / "Live" / suffix / "drawer.txt"
    wrong.parent.mkdir(parents=True)
    wrong.write_bytes(raw.read_bytes())
    payload["captures"][0]["raw_text_path"] = str(wrong)
    bundle.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ingest.PluginIngestError, match="outside workspace"):
        ingest.ingest_capture_bundle(bundle, workspace_root=tmp_path,
            live_root=tmp_path / "Live", audit_dir=tmp_path / "outputs/audits/plugin_cycles")


def _drawer_text(
    *,
    symbol: str = "BTCUSDT",
    strategy_id: str = "413500001",
    created_at_lima: str = "2026-08-13 10:00:00",
) -> str:
    return "\n".join(
        [
            symbol,
            "Perp",
            "Futures Grid",
            "Working",
            "Time Created",
            created_at_lima,
            "Total Profit (USDT)",
            "1.23 (1.23%)",
            "Invested Margin (USDT)",
            "100.00",
            "Matched Profit (USDT)",
            "1.00 (1.00%)",
            "Unmatched PNL (USDT)",
            "0.23 (0.23%)",
            "Funding Fee (USDT)",
            "0.00 (0.00%)",
            "Positions",
            "no position",
            "Pending Order",
            "Grid Details",
            "Mode",
            "Geometric",
            "Price Range",
            "1.0 - 2.0 USDT",
            "Number of Grids",
            "10",
            "Initial Leverage",
            "10x",
            "Order History",
            "Strategy Number",
            strategy_id,
        ]
    )


def _write_bundle(
    workspace: Path,
    *,
    symbol: str = "BTCUSDT",
    strategy_id: str = "413500001",
    raw_text: str | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    staging = workspace / "outputs" / "runtime" / "chrome_plugin_capture" / "run-1"
    staging.mkdir(parents=True)
    raw_path = staging / f"{symbol}.txt"
    text = raw_text or _drawer_text(symbol=symbol, strategy_id=strategy_id)
    raw_path.write_text(text, encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": ingest.PLUGIN_CAPTURE_BUNDLE_SCHEMA,
        "status": "complete",
        "source": "chrome_plugin",
        "run_id": "run-1",
        "page_identity": "Binance USD-M Futures Grid",
        "source_url": "https://www.binance.com/en/futures/grid",
        "authenticated": True,
        "cycle_started_at_utc": "2026-08-13T15:00:00+00:00",
        "cycle_completed_at_utc": "2026-08-13T15:00:30+00:00",
        "working_row_count": 1,
        "roster_before_symbols": [symbol],
        "roster_after_symbols": [symbol],
        "captures": [
            {
                "symbol": symbol,
                "strategy_id": strategy_id,
                "working_status": "Working",
                "deployment_time_lima": "2026-08-13 10:00:00",
                "captured_at_utc": "2026-08-13T15:00:15+00:00",
                "raw_text_path": str(raw_path),
                "raw_text_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                "capture_status": "complete",
            }
        ],
    }
    bundle_path = staging / "capture_bundle.json"
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    return bundle_path, raw_path, payload


def test_ingest_plugin_bundle_commits_strict_cycle_and_controller_loads_it(
    tmp_path: Path,
) -> None:
    bundle_path, raw_path, _ = _write_bundle(tmp_path)
    live_root = tmp_path / "Live"
    audit_dir = tmp_path / "outputs" / "audits" / "plugin_cycles"

    cycle_path = ingest.ingest_capture_bundle(
        bundle_path,
        workspace_root=tmp_path,
        live_root=live_root,
        audit_dir=audit_dir,
    )

    cycle_payload = json.loads(cycle_path.read_text(encoding="utf-8"))
    assert cycle_payload["schema_version"] == controller.PLUGIN_CYCLE_SCHEMA
    assert cycle_payload["source"] == "chrome_plugin"
    assert cycle_payload["status"] == "complete"
    targets_path = Path(cycle_payload["collector_targets_csv"])
    assert targets_path.read_text(encoding="utf-8").splitlines() == [
        "symbol,strategy_id,deploy_ts",
        "BTCUSDT,413500001,2026-08-13T15:00:00+00:00",
    ]
    assert Path(cycle_payload["files"][0]["text_path"]) != raw_path
    metadata_path = Path(cycle_payload["files"][0]["json_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == controller.PLUGIN_SNAPSHOT_SCHEMA
    assert metadata["source"] == "chrome_plugin"
    assert metadata["raw_text_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert metadata["structured_telemetry"]["deploy_ts"] == (
        "2026-08-13T15:00:00+00:00"
    )

    cycle = controller.load_complete_cycle(
        cycle_path,
        allowed_live_root=live_root,
    )
    assert cycle.symbols == ("BTCUSDT",)
    assert cycle.strategy_ids == ("413500001",)
    assert cycle.bots[0].scanner_entry["execution_telemetry"]["source"] == (
        "chrome_plugin"
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(roster_after_symbols=["ETHUSDT"]),
            "roster changed",
        ),
        (
            lambda payload: payload["captures"][0].update(raw_text_sha256="0" * 64),
            "SHA-256 mismatch",
        ),
        (
            lambda payload: payload["captures"][0].update(
                deployment_time_lima="2026-08-13 09:59:59"
            ),
            "deployment time mismatch",
        ),
    ],
)
def test_ingest_plugin_bundle_rejects_identity_or_integrity_mismatch(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    bundle_path, _raw_path, payload = _write_bundle(tmp_path)
    assert callable(mutation)
    mutation(payload)
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ingest.PluginIngestError, match=message):
        ingest.ingest_capture_bundle(
            bundle_path,
            workspace_root=tmp_path,
            live_root=tmp_path / "Live",
            audit_dir=tmp_path / "outputs" / "audits" / "plugin_cycles",
        )


def test_ingest_plugin_bundle_rejects_nonfinite_json(tmp_path: Path) -> None:
    bundle_path, _raw_path, _payload = _write_bundle(tmp_path)
    bundle_path.write_text('{"schema_version": NaN}', encoding="utf-8")

    with pytest.raises(ingest.PluginIngestError, match="non-finite JSON"):
        ingest.ingest_capture_bundle(
            bundle_path,
            workspace_root=tmp_path,
            live_root=tmp_path / "Live",
            audit_dir=tmp_path / "outputs" / "audits" / "plugin_cycles",
        )


def test_ingest_plugin_bundle_accepts_binance_bahrain_hostname(
    tmp_path: Path,
) -> None:
    bundle_path, _raw_path, payload = _write_bundle(tmp_path)
    payload["source_url"] = "https://www.binance.bh/en/trading-bots/futures/grid/BTCUSDT"
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")

    cycle_path = ingest.ingest_capture_bundle(
        bundle_path,
        workspace_root=tmp_path,
        live_root=tmp_path / "Live",
        audit_dir=tmp_path / "outputs" / "audits" / "plugin_cycles",
    )

    cycle = json.loads(cycle_path.read_text(encoding="utf-8"))
    assert cycle["source_url"] == payload["source_url"]


@pytest.mark.parametrize(
    "source_url",
    [
        "https://not-binance.example/futures/grid",
        "https://www.binance.com.evil.example/futures/grid",
        "https://www.binance.bh.evil.example/futures/grid",
        "https://binance-bh.example/futures/grid",
        "http://www.binance.bh/futures/grid",
    ],
)
def test_ingest_plugin_bundle_rejects_untrusted_binance_hostname(
    tmp_path: Path,
    source_url: str,
) -> None:
    bundle_path, _raw_path, payload = _write_bundle(tmp_path)
    payload["source_url"] = source_url
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ingest.PluginIngestError, match="HTTPS Binance page"):
        ingest.ingest_capture_bundle(
            bundle_path,
            workspace_root=tmp_path,
            live_root=tmp_path / "Live",
            audit_dir=tmp_path / "outputs" / "audits" / "plugin_cycles",
        )


def test_ingest_plugin_bundle_rejects_capture_outside_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    bundle_path, _raw_path, payload = _write_bundle(workspace)
    outside = tmp_path / "outside.txt"
    outside.write_text(_drawer_text(), encoding="utf-8")
    capture = payload["captures"][0]
    assert isinstance(capture, dict)
    capture["raw_text_path"] = str(outside)
    capture["raw_text_sha256"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ingest.PluginIngestError, match="outside workspace"):
        ingest.ingest_capture_bundle(
            bundle_path,
            workspace_root=workspace,
            live_root=workspace / "Live",
            audit_dir=workspace / "outputs" / "audits" / "plugin_cycles",
        )


def test_plugin_cycle_hash_tampering_is_rejected(tmp_path: Path) -> None:
    bundle_path, _raw_path, _payload = _write_bundle(tmp_path)
    live_root = tmp_path / "Live"
    cycle_path = ingest.ingest_capture_bundle(
        bundle_path,
        workspace_root=tmp_path,
        live_root=live_root,
        audit_dir=tmp_path / "outputs" / "audits" / "plugin_cycles",
    )
    cycle_payload = json.loads(cycle_path.read_text(encoding="utf-8"))
    committed_raw = Path(cycle_payload["files"][0]["text_path"])
    committed_raw.write_text("tampered", encoding="utf-8")

    with pytest.raises(controller.ControllerError, match="SHA-256 mismatch"):
        controller.load_complete_cycle(cycle_path, allowed_live_root=live_root)


def test_plugin_acquisition_never_calls_cdp_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path, _raw_path, _payload = _write_bundle(tmp_path)
    live_root = tmp_path / "Live"
    cycle_path = ingest.ingest_capture_bundle(
        bundle_path,
        workspace_root=tmp_path,
        live_root=live_root,
        audit_dir=tmp_path / "outputs" / "audits" / "plugin_cycles",
    )
    args = type(
        "Args",
        (),
        {
            "acquisition_mode": "plugin-manifest",
            "cycle_manifest": cycle_path,
            "live_root": live_root,
            "max_telemetry_age_seconds": 900.0,
        },
    )()

    def forbidden_fetch(_args: object) -> controller.TelemetryCycle:
        raise AssertionError("CDP fetch must not be called")

    monkeypatch.setattr(controller, "fetch_private_cycle", forbidden_fetch)
    loaded, source = controller.acquire_cycle(
        args,
        now=datetime(2026, 8, 13, 15, 1, tzinfo=UTC),
    )

    assert source == "chrome_plugin_manifest"
    assert loaded.manifest_path == cycle_path.resolve()


def test_ingest_plugin_bundle_rejects_duplicate_strategy_ids(tmp_path: Path) -> None:
    bundle_path, _raw_path, payload = _write_bundle(tmp_path)
    staging = bundle_path.parent
    second_text = _drawer_text(symbol="ETHUSDT", strategy_id="413500001")
    second_path = staging / "ETHUSDT.txt"
    second_path.write_text(second_text, encoding="utf-8")
    payload["working_row_count"] = 2
    payload["roster_before_symbols"] = ["BTCUSDT", "ETHUSDT"]
    payload["roster_after_symbols"] = ["BTCUSDT", "ETHUSDT"]
    captures = payload["captures"]
    assert isinstance(captures, list)
    captures.append(
        {
            "symbol": "ETHUSDT",
            "strategy_id": "413500001",
            "working_status": "Working",
            "deployment_time_lima": "2026-08-13 10:00:00",
            "captured_at_utc": "2026-08-13T15:00:20+00:00",
            "raw_text_path": str(second_path),
            "raw_text_sha256": hashlib.sha256(second_path.read_bytes()).hexdigest(),
            "capture_status": "complete",
        }
    )
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ingest.PluginIngestError, match="strategy IDs are duplicated"):
        ingest.ingest_capture_bundle(
            bundle_path,
            workspace_root=tmp_path,
            live_root=tmp_path / "Live",
            audit_dir=tmp_path / "outputs" / "audits" / "plugin_cycles",
        )


def test_plugin_cycle_validates_optional_screenshot_hash(tmp_path: Path) -> None:
    bundle_path, _raw_path, payload = _write_bundle(tmp_path)
    screenshot = bundle_path.parent / "BTCUSDT.png"
    screenshot.write_bytes(b"not-a-real-image-but-immutable-evidence")
    capture = payload["captures"][0]
    assert isinstance(capture, dict)
    capture["screenshot_path"] = str(screenshot)
    capture["screenshot_sha256"] = hashlib.sha256(screenshot.read_bytes()).hexdigest()
    bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    live_root = tmp_path / "Live"
    cycle_path = ingest.ingest_capture_bundle(
        bundle_path,
        workspace_root=tmp_path,
        live_root=live_root,
        audit_dir=tmp_path / "outputs" / "audits" / "plugin_cycles",
    )
    payload = json.loads(cycle_path.read_text(encoding="utf-8"))
    committed_screenshot = Path(payload["files"][0]["screenshot_path"])
    controller.load_complete_cycle(cycle_path, allowed_live_root=live_root)
    committed_screenshot.write_bytes(b"tampered")

    with pytest.raises(controller.ControllerError, match="screenshot SHA-256 mismatch"):
        controller.load_complete_cycle(cycle_path, allowed_live_root=live_root)


def test_plugin_cycle_rejects_tampered_collector_targets(tmp_path: Path) -> None:
    bundle_path, _raw_path, _payload = _write_bundle(tmp_path)
    live_root = tmp_path / "Live"
    cycle_path = ingest.ingest_capture_bundle(
        bundle_path,
        workspace_root=tmp_path,
        live_root=live_root,
        audit_dir=tmp_path / "outputs" / "audits" / "plugin_cycles",
    )
    cycle_payload = json.loads(cycle_path.read_text(encoding="utf-8"))
    targets_path = Path(cycle_payload["collector_targets_csv"])
    targets_path.write_text("symbol,strategy_id\nETHUSDT,wrong\n", encoding="utf-8")

    with pytest.raises(controller.ControllerError, match="targets SHA-256 mismatch"):
        controller.load_complete_cycle(cycle_path, allowed_live_root=live_root)
