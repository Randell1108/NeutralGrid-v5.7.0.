# GRIDFIX-001 §2.6 — distribution diff smoke (2026-05-04 16:56 UTC)

Compares the new mode-aware `profit_per_grid_pct` (post-§2.1+§2.3) against the
pre-fix silent-geometric value. Source: `data/new_expired_bots.xlsx` (canonical).

## Coverage

- Total rows: 209
- Mode counts (canonical): arithmetic=166 geometric=43
- Mapper rows after exclusion gate: 209 (zero dropped means §2.5 inert today)
- new PPG populated: 209/209
- silent-flip PPG populated: 209/209
- both populated: 209 | new only: 0 | silent only: 0 | both NaN: 0

## Per-mode delta (new - silent_flip), bps

### ARITHMETIC rows  (n=166)

- mean: 1.0433 bps
- median: 0.0842 bps
- std: 4.4962 bps
- min: -0.0187 bps
- max: 38.4725 bps
- p5: -0.0085 bps
- p95: 3.1959 bps

### GEOMETRIC rows  (n=43)

- mean: 0.0000 bps
- median: 0.0000 bps
- std: 0.0000 bps
- min: 0.0000 bps
- max: 0.0000 bps
- p5: 0.0000 bps
- p95: 0.0000 bps

## Distribution of `new_ppg` (post-fix), all rows

- n=209
- mean=1.0025%
- median=0.9479%
- std=0.4121%
- p5=0.5589%
- p95=1.7485%

## Interpretation

- ARITHMETIC rows: nonzero delta = silent-flip bug correction. Magnitude is the wrong-mode error that was being silently injected pre-fix.
- GEOMETRIC rows: delta should be ~zero (geometric == geometric). Any non-zero indicates a residual bug.
- A positive delta means the new (correct) PPG is higher than the silent-flip value.
