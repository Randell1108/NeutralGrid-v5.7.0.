# Prompt to run after operational prompt line 37

Copy the following paragraph into this repository's Codex task after executing
the operational prompt's P0 start-and-verify commands.

> Run the recurring live-drawer verdict workflow in `D:\Neutral Grid v5.7.0` using
> `docs/recurring_drawer_verdict.md`. Use my authenticated Chrome extension to
> capture every currently Working UM Grid bot, prove the complete roster before
> and after capture, save raw data under `Live/<Lima-date>/<SYMBOL>/`, ingest the
> returned bundle, and consume the exact resulting manifest once through
> `scripts/run_drawer_verdict_cycle.py` with the current P0 public L2 manifest.
> Apply the read-only artifact preflight and preserve all source failures and
> warnings. Use the controller's default 10-minute cadence and 15-minute drawer
> age limit. Create or update the existing recurring Codex task in this
> conversation; do not create a duplicate schedule. Keep the P0 market services
> running and retire the superseded dedicated-profile drawer loop through its
> stop script if it is still running. Check `--status` before reporting a verdict.
> Return CONTINUE, ADJUST or END only as the scanner's current advisory result,
> with the exact symbol/strategy, timestamp, reasons and evidence paths. Record
> any acquisition failure with `--capture-failure`; stale or incomplete evidence
> must produce no current verdict. Do not execute trading actions, send Discord
> messages, modify thresholds, retrain models or admit live data into a training
> pool. Remain quiet when the state is unchanged; notify me of a meaningful
> verdict/roster change, failure, staleness, recovery or required user action.
