from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from neutralgrid.live.decision.native_sar import NativeRecord
from neutralgrid.live.decision.state_action_reward import SarError, SourceRef, file_hash
from neutralgrid.live.decision import terminal_outcomes as terminal


TEXT = "\n".join([
    "FARTCOINUSDT", "Perp", "Futures Grid", "Canceled", "Time Ended", "2026-10-22 00:22:32",
    "Total Profit (USDT)", "-56.32 (-5.63%)", "Invested Margin (USDT)", "1,000.00",
    "Strategy Number", "415222014", "Time Created", "2026-10-21 19:22:22",
])


def original(tmp_path: Path, *, text: str = TEXT, date: str = "2026-10-25", **metadata) -> tuple[Path, NativeRecord]:
    live = tmp_path / "Live"
    path = live / date / "FARTCOINUSDT" / "verdict_audit_415222014" / "strategy_detail.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = {"text": text, "captured_at_utc": "2026-10-25T15:51:27.985Z", "display_timezone": "UTC-5"}
    raw.update(metadata)
    path.write_text(json.dumps(raw), encoding="utf-8")
    return live, NativeRecord(SourceRef(path=str(path), sha256=file_hash(path)))


def cli():
    path = Path(__file__).resolve().parents[2] / "scripts/capture_terminal_bot_outcomes.py"
    spec = importlib.util.spec_from_file_location("terminal_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_roundtrip_scores_reported_capital_deficit_without_reward(tmp_path):
    live, native = original(tmp_path)
    path, created = terminal.capture_terminal_outcome(native, live_root=live, ingested_at=datetime(2026, 10, 26, tzinfo=timezone.utc))
    assert created and path.parts[-4] == "2026-10-25"
    observation = terminal.load_terminal_outcome(path, live_root=live)
    assert observation.bot.deployed_at.isoformat() == "2026-10-22T00:22:22+00:00"
    assert observation.ended_at.isoformat() == "2026-10-22T05:22:32+00:00"
    score = terminal.score_terminal_outcome(observation, live_root=live)
    assert Decimal(score["reported_capital_deficit_usdt"]) == Decimal("56.32")
    assert Decimal(score["normalized_return_pct"]) == Decimal("-5.632")
    assert score["displayed_return_pct"] == "-5.63"
    assert score["first_passage_status"] == "path_unobserved"
    assert score["observed_action"] is score["causal_reward"] is None
    assert not any(score[k] for k in ("verified_equity", "training_eligible", "promotion_eligible", "influence_enabled"))


@pytest.mark.parametrize("pnl,expected", [("3.88 (0.38%)", False), ("30.00 (3.00%)", True), ("0 (0%)", False)])
def test_positive_flat_and_terminal_target_witness(tmp_path, pnl, expected):
    live, native = original(tmp_path, text=TEXT.replace("-56.32 (-5.63%)", pnl))
    score = terminal.score_terminal_outcome(terminal.parse_terminal_capture(native, live_root=live), live_root=live)
    assert score["first_passage_status"] == 'path_unobserved'
    assert score['terminal_return_meets_3pct_before_deployment_deadline'] is expected
    assert score['capital_basis_at_deployment_verified'] is False
    assert Decimal(score["reported_capital_deficit_usdt"]) == 0


def test_after_seven_hours_positive_endpoint_does_not_prove_target_within_horizon(tmp_path):
    live, native = original(tmp_path, text=TEXT.replace("-56.32 (-5.63%)", "40 (4%)").replace("2026-10-22 00:22:32", "2026-10-22 03:22:32"))
    score = terminal.score_terminal_outcome(terminal.parse_terminal_capture(native, live_root=live), live_root=live)
    assert score["first_passage_status"] == "path_unobserved"


def test_capture_timestamp_alias_accepted_but_conflict_rejected(tmp_path):
    live, native = original(tmp_path, captured_at="2026-10-25T15:51:27.985Z")
    raw = json.loads(Path(native.source.path).read_text())
    del raw["captured_at_utc"]
    path = Path(native.source.path)
    path.write_text(json.dumps(raw))
    native = NativeRecord(SourceRef(path=str(path), sha256=file_hash(path)))
    assert terminal.parse_terminal_capture(native, live_root=live).captured_at.hour == 15
    live, native = original(tmp_path, captured_at="2026-10-25T15:52:27.985Z")
    with pytest.raises(SarError, match="conflicting capture"):
        terminal.parse_terminal_capture(native, live_root=live)


@pytest.mark.parametrize("metadata", [
    {"display_timezone": "UTC"}, {"display_timezone": None}, {"captured_at_utc": "2026-10-25T15:51:27"},
    {"captured_at_utc": None}, {"strategy_id": "999"}, {"strategy_id": 415222014},
    {"symbol": "ARKMUSDT"}, {"source": "manual transcription"}, {"text": None}, {"status": "Working"},
])
def test_invalid_metadata_fails_closed(tmp_path, metadata):
    live, native = original(tmp_path, **metadata)
    with pytest.raises((ValueError, SarError)):
        terminal.parse_terminal_capture(native, live_root=live)


@pytest.mark.parametrize("text", [
    TEXT.replace("Canceled", "Working"), TEXT.replace("Strategy Number", "Missing Number"),
    TEXT + "\nStrategy Number\n415222014", TEXT + "\nTotal Profit (USDT)\n1 (1%)",
    TEXT.replace("1,000.00", "0"), TEXT.replace("1,000.00", "-1"), TEXT.replace("1,000.00", "1,00"),
    TEXT.replace("-56.32 (-5.63%)", "NaN (NaN%)"), TEXT.replace("-56.32 (-5.63%)", "Infinity (Infinity%)"),
    TEXT.replace("2026-10-22 00:22:32", "2026-10-20 00:22:32"),
    TEXT.replace("2026-10-22 00:22:32", "2026-10-26 00:22:32"),
    TEXT.replace("2026-10-21 19:22:22", "2026-10-21T19:22:22"),
])
def test_native_field_and_timing_failures(tmp_path, text):
    live, native = original(tmp_path, text=text)
    with pytest.raises((ValueError, SarError)):
        terminal.parse_terminal_capture(native, live_root=live)


def test_duplicate_json_keys_and_nonfinite_json_rejected(tmp_path):
    live, native = original(tmp_path)
    path = Path(native.source.path)
    for invalid in ('{"text":"a","text":"b"}', '{"amount":NaN}'):
        path.write_text(invalid)
        current = NativeRecord(SourceRef(path=str(path), sha256=file_hash(path)))
        with pytest.raises(ValueError):
            terminal.parse_terminal_capture(current, live_root=live)


def test_idempotent_recapture_cross_date_preserves_first_provenance(tmp_path):
    live, first = original(tmp_path)
    path, _ = terminal.capture_terminal_outcome(first, live_root=live)
    before = path.read_bytes()
    _, later = original(tmp_path, date="2026-10-26", captured_at_utc="2026-10-26T15:00:00Z")
    same, created = terminal.capture_terminal_outcome(later, live_root=live, ingested_at=datetime(2026, 10, 27, tzinfo=timezone.utc))
    assert same == path and not created and path.read_bytes() == before
    assert terminal.load_terminal_outcome(path, live_root=live).source == first.source


@pytest.mark.parametrize("replacement", ["-50.00 (-5.00%)", "-56.32 (-5.64%)"])
def test_conflicting_recapture_cannot_overwrite(tmp_path, replacement):
    live, first = original(tmp_path)
    path, _ = terminal.capture_terminal_outcome(first, live_root=live)
    before = path.read_bytes()
    _, second = original(tmp_path, date="2026-10-26", text=TEXT.replace("-56.32 (-5.63%)", replacement))
    with pytest.raises(terminal.TerminalStorageError, match="conflicting terminal"):
        terminal.capture_terminal_outcome(second, live_root=live)
    assert path.read_bytes() == before


def test_original_mutation_and_stored_tampering_rejected(tmp_path):
    live, native = original(tmp_path)
    path, _ = terminal.capture_terminal_outcome(native, live_root=live)
    stored = path.read_bytes()
    raw = json.loads(stored)
    raw['reported_total_pnl_usdt'] = '10'
    path.write_text(json.dumps(raw))
    with pytest.raises(SarError, match="differs"):
        terminal.load_terminal_outcome(path, live_root=live)
    path.write_bytes(stored)
    Path(native.source.path).write_text('{}')
    with pytest.raises(SarError, match="changed"):
        terminal.load_terminal_outcome(path, live_root=live)


def test_duplicate_paths_rejected(tmp_path):
    live, native = original(tmp_path)
    path, _ = terminal.capture_terminal_outcome(native, live_root=live)
    duplicate = live / '2026-10-29' / path.parts[-3] / path.parts[-2] / path.name
    duplicate.parent.mkdir(parents=True)
    shutil.copyfile(path, duplicate)
    with pytest.raises(terminal.TerminalStorageError, match="duplicate terminal"):
        terminal.capture_terminal_outcome(native, live_root=live)


def test_wrong_root_and_route_rejected(tmp_path):
    live, native = original(tmp_path)
    with pytest.raises(SarError, match="inside"):
        terminal.parse_terminal_capture(native, live_root=tmp_path / 'other' / 'Live')
    with pytest.raises(SarError, match="JSONL"):
        terminal.parse_terminal_capture(NativeRecord(native.source, 1), live_root=live)
    outside = live / '2026-10-25' / 'FARTCOINUSDT' / 'wrong' / 'strategy_detail.json'
    outside.parent.mkdir()
    shutil.copyfile(native.source.path, outside)
    with pytest.raises(SarError, match="path"):
        terminal.parse_terminal_capture(NativeRecord(SourceRef(path=str(outside),sha256=file_hash(outside))), live_root=live)


def test_publication_then_source_failure_exposes_published_path(tmp_path, monkeypatch):
    live, native = original(tmp_path)
    publish = terminal.atomic_create_json
    def publish_then_mutate(path, payload):
        publish(path, payload)
        Path(native.source.path).write_text('{}')
    monkeypatch.setattr(terminal, 'atomic_create_json', publish_then_mutate)
    with pytest.raises(terminal.TerminalStorageError) as error:
        terminal.capture_terminal_outcome(native, live_root=live)
    assert error.value.published_path is not None and error.value.published_path.exists()


def test_cli_default_report_only_and_partial_capture(tmp_path):
    live, native = original(tmp_path)
    module = cli()
    report = module.run([Path(native.source.path)], output_dir=tmp_path/'reports'/'inspect', root=tmp_path)
    assert report['mode'] == 'report_only' and report['status'] == 'completed'
    assert not list(live.glob('*/FARTCOINUSDT/terminal_outcomes/*.json'))
    result = module.run([Path(native.source.path), live/'missing.json'], output_dir=tmp_path/'reports'/'capture', root=tmp_path, capture=True)
    assert result['status'] == 'partial_failure'
    assert len(result['results']) == 1 and len(result['errors']) == 1
    assert Path(result['results'][0]['stored_path']).exists()
    assert (tmp_path/'reports'/'capture'/'report.json').exists()
    with pytest.raises(ValueError, match="fresh"):
        module.run([Path(native.source.path)], output_dir=tmp_path/'reports'/'capture', root=tmp_path)
    with pytest.raises(ValueError, match="reports"):
        module.run([Path(native.source.path)], output_dir=tmp_path/'outside', root=tmp_path)


def test_cli_checkpoint_failure_reports_published_path(tmp_path, monkeypatch):
    _, native = original(tmp_path)
    module = cli()
    def fail(*args):
        raise OSError('disk failure')
    monkeypatch.setattr(module, 'atomic_create_json', fail)
    with pytest.raises(terminal.TerminalStorageError) as error:
        module.run([Path(native.source.path)], output_dir=tmp_path/'reports'/'capture', root=tmp_path, capture=True)
    assert error.value.published_path is not None and error.value.published_path.exists()


def test_storage_failure_before_publication_has_no_outcome(tmp_path, monkeypatch):
    live, native = original(tmp_path)
    def fail(*args):
        raise OSError('storage unavailable')
    monkeypatch.setattr(terminal, 'atomic_create_json', fail)
    with pytest.raises(terminal.TerminalStorageError) as error:
        terminal.capture_terminal_outcome(native, live_root=live)
    assert error.value.published_path is None
    assert not list(live.glob('*/FARTCOINUSDT/terminal_outcomes/*.json'))


def test_arkm_native_header_without_declared_identity(tmp_path):
    live, _ = original(tmp_path)
    path = live/'2026-10-25'/'ARKMUSDT'/'verdict_audit_415232572'/'strategy_detail.json'
    path.parent.mkdir(parents=True)
    text = TEXT.replace('FARTCOINUSDT','ARKMUSDT').replace('415222014','415232572').replace('-56.32 (-5.63%)','3.88 (0.38%)').replace('2026-10-21 19:22:22','2026-10-22 08:49:00').replace('2026-10-22 00:22:32','2026-10-22 09:23:10')
    path.write_text(json.dumps({'text':text,'captured_at':'2026-10-25T15:46:51.226Z','display_timezone':'UTC-5'}))
    native = NativeRecord(SourceRef(path=str(path),sha256=file_hash(path)))
    observation = terminal.parse_terminal_capture(native,live_root=live)
    score = terminal.score_terminal_outcome(observation,live_root=live)
    assert observation.bot.strategy_id == '415232572'
    assert Decimal(score['normalized_return_pct']) == Decimal('0.388')
    assert observation.ended_at.isoformat() == '2026-10-22T14:23:10+00:00'


def test_direct_model_tampering_cannot_enable_score(tmp_path):
    live, native = original(tmp_path)
    observation = terminal.parse_terminal_capture(native,live_root=live)
    forged = observation.model_copy(update={'training_eligible':True})
    with pytest.raises(SarError, match='not proved'):
        terminal.score_terminal_outcome(forged,live_root=live)


def test_cli_main_malformed_source_and_invalid_output_are_controlled(tmp_path,monkeypatch):
    _, native = original(tmp_path)
    Path(native.source.path).write_text('{broken')
    module = cli()
    actual_run = module.run
    monkeypatch.setattr(module,'run',lambda sources,**kwargs:actual_run(sources,root=tmp_path,**kwargs))
    assert module.main(['--source',native.source.path,'--output-dir',str(tmp_path/'reports'/'bad')]) == 1
    assert module.main(['--source',native.source.path,'--output-dir',str(tmp_path/'not_reports')]) == 2


def test_terminal_audit_scores_once_and_reports_source_refs(tmp_path):
    live, native = original(tmp_path)
    path, _ = terminal.capture_terminal_outcome(native,live_root=live)
    report = terminal.audit_terminal_outcomes(live)
    assert report['accepted_terminal_outcomes'] == 1
    assert report['rejected'] == []
    assert report['outcome_classes'] == {'negative':1}
    assert report['scores'][0]['terminal_outcome_source']['path'] == str(path)
    assert report['scores'][0]['source'] == native.source.model_dump(mode='json')
    assert len(report['code_sha256']) == 64


@pytest.mark.parametrize('corrupt', [False,True])
def test_terminal_audit_quarantines_all_duplicates_even_if_one_unparseable(tmp_path,corrupt):
    live, native = original(tmp_path)
    path, _ = terminal.capture_terminal_outcome(native,live_root=live)
    duplicate = live/'not-a-date'/'FARTCOINUSDT'/'terminal_outcomes'/path.name
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b'{broken' if corrupt else path.read_bytes())
    report = terminal.audit_terminal_outcomes(live)
    assert report['accepted_terminal_outcomes'] == 0
    assert report['scores'] == []
    assert report['rejected_terminal_records'] == 2
    assert path.stem in report['quarantined_bot_identity_keys']


def test_terminal_audit_misnamed_valid_identity_quarantines_legitimate_copy(tmp_path):
    live, native = original(tmp_path)
    path, _ = terminal.capture_terminal_outcome(native,live_root=live)
    duplicate = path.with_name('malformed-name.json')
    duplicate.write_bytes(path.read_bytes())
    report = terminal.audit_terminal_outcomes(live)
    assert report['accepted_terminal_outcomes'] == 0
    assert len(report['rejected']) == 2


def test_terminal_audit_rejects_mutated_original_and_unknown_malformed_file(tmp_path):
    live, native = original(tmp_path)
    path, _ = terminal.capture_terminal_outcome(native,live_root=live)
    Path(native.source.path).write_text('{}')
    path.with_name('unknown.json').write_text('{bad')
    report = terminal.audit_terminal_outcomes(live)
    assert report['scores'] == []
    assert report['rejected_terminal_records'] == 2
    assert any(r['path'].endswith('unknown.json') and not r['bot_identity_keys'] for r in report['rejected'])


def test_multiple_native_headers_rejected_even_with_single_economic_section(tmp_path):
    live, native = original(tmp_path,text=TEXT+'\nFARTCOINUSDT\nPerp\nFutures Grid\nWorking')
    with pytest.raises(SarError,match='multiple native'):
        terminal.parse_terminal_capture(native,live_root=live)


def test_ingestion_cannot_precede_capture_and_writes_nothing(tmp_path):
    live, native = original(tmp_path)
    with pytest.raises(SarError,match='ingestion cutoff'):
        terminal.capture_terminal_outcome(native,live_root=live,ingested_at=datetime(2026,10,25,tzinfo=timezone.utc))
    assert not (live/'.terminal-outcome-locks').exists()
    assert not list(live.glob('*/FARTCOINUSDT/terminal_outcomes/*.json'))


@pytest.mark.parametrize('capture',[False,True])
def test_batch_conflicts_quarantine_every_score_before_terminal_writes(tmp_path,capture):
    live, first = original(tmp_path)
    _, second = original(tmp_path,date='2026-10-24',text=TEXT.replace('-56.32 (-5.63%)','-50 (-5%)'))
    report = cli().run([Path(first.source.path),Path(second.source.path)],output_dir=tmp_path/'reports'/'conflict',root=tmp_path,capture=capture)
    assert report['results']==[] and len(report['errors'])==2
    assert report['status']=='failed'
    assert not list(live.glob('*/FARTCOINUSDT/terminal_outcomes/*.json'))


def test_batch_repeated_path_and_recapture_scored_once_earliest_provenance(tmp_path):
    live, later = original(tmp_path)
    _, earlier = original(tmp_path,date='2026-10-24',captured_at_utc='2026-10-24T15:00:00Z')
    report = cli().run([Path(later.source.path),Path(earlier.source.path),Path(earlier.source.path)],output_dir=tmp_path/'reports'/'dedup',root=tmp_path,capture=True)
    assert len(report['results'])==1 and report['duplicate_source_count']==2
    assert report['results'][0]['source_path']==earlier.source.path
    stored=terminal.load_terminal_outcome(Path(report['results'][0]['stored_path']),live_root=live)
    assert stored.source==earlier.source


def test_extreme_numeric_token_rejected_without_decimal_overflow(tmp_path):
    live,native=original(tmp_path,text=TEXT.replace('1,000.00','0.'+'0'*1000001+'1'))
    with pytest.raises(SarError,match='bounded input'):
        terminal.parse_terminal_capture(native,live_root=live)


def test_changed_terminal_margin_cannot_establish_fastwin_capital_basis(tmp_path):
    live,native=original(tmp_path,text=TEXT.replace('-56.32 (-5.63%)','30 (6%)').replace('1,000.00','500'))
    score=terminal.score_terminal_outcome(terminal.parse_terminal_capture(native,live_root=live),live_root=live)
    assert score['terminal_return_meets_3pct_before_deployment_deadline'] is True
    assert score['first_passage_status']=='path_unobserved'
    assert score['capital_basis_at_deployment_verified'] is False


def test_backdated_terminal_record_is_rejected_and_quarantined(tmp_path):
    live, native = original(tmp_path)
    path, _ = terminal.capture_terminal_outcome(native, live_root=live)
    backdated = live/'2026-10-24'/'FARTCOINUSDT'/'terminal_outcomes'/path.name
    backdated.parent.mkdir(parents=True)
    path.replace(backdated)
    with pytest.raises(SarError, match='folder predates'):
        terminal.load_terminal_outcome(backdated, live_root=live)
    report = terminal.audit_terminal_outcomes(live)
    assert report['scores'] == []
    assert report['rejected_terminal_records'] == 1
    assert backdated.stem in report['quarantined_bot_identity_keys']
