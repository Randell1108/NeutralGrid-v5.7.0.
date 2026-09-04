"""Display scanner candidates with enriched grid parameters."""
import pandas as pd
from pathlib import Path

# Auto-detect latest results file
results_dir = Path('results')
csv_files = sorted(results_dir.glob('neutralgrid_candidates_*.csv'))
if not csv_files:
    print("No neutralgrid_candidates_*.csv files found in results/")
    raise SystemExit(1)
df = pd.read_csv(csv_files[-1])
df_sorted = df.sort_values('score', ascending=False)

print('=' * 120)
print('NEUTRAL GRID SCANNER RESULTS - TOP 20 CANDIDATES WITH ENRICHED PARAMETERS')
print('=' * 120)
print(f'Scan File: {csv_files[-1].name}')
print(f'Total Symbols Scanned: {len(df)}')
print(f'Score Range: {df["score"].min():.2f} - {df["score"].max():.2f}')
print()

# Filter for most promising candidates (score > 40)
promising = df_sorted[df_sorted['score'] >= 40].head(20)
print(f'Candidates with Score >= 40: {len(df_sorted[df_sorted["score"] >= 40])}')
print()

print('-' * 120)
print(f'{"Symbol":<15} {"Score":>6} {"Grids":>6} {"Profit/Grid":>11} {"Range Prob":>10} {"Trend Prob":>10} {"Survival":>9} {"OU Half":>8} {"ADX 1h":>7} {"RSI 15m":>8}')
print('-' * 120)

for idx, row in promising.iterrows():
    symbol = row['symbol']
    score = row['score']
    num_grids = int(row['num_grids']) if pd.notna(row.get('num_grids')) else '-'
    profit_grid = f"{row['profit_per_grid_pct']:.2f}%" if pd.notna(row.get('profit_per_grid_pct')) else '-'
    range_prob = f"{row['hmm_range_prob']:.3f}" if pd.notna(row.get('hmm_range_prob')) else '-'
    trend_prob = f"{row['hmm_trend_prob']:.3f}" if pd.notna(row.get('hmm_trend_prob')) else '-'
    survival = f"{row['survival_prob']:.2f}" if pd.notna(row.get('survival_prob')) else '-'
    ou_half = f"{row['ou_halflife']:.1f}" if pd.notna(row.get('ou_halflife')) else '-'
    adx_1h = f"{row['adx_1h']:.1f}" if pd.notna(row.get('adx_1h')) else '-'
    rsi_15m = f"{row['rsi_15m']:.1f}" if pd.notna(row.get('rsi_15m')) else '-'

    print(f'{symbol:<15} {score:>6.2f} {str(num_grids):>6} {profit_grid:>11} {range_prob:>10} {trend_prob:>10} {survival:>9} {ou_half:>8} {adx_1h:>7} {rsi_15m:>8}')

print()
print('=' * 120)
print('DETAILED VIEW - TOP 10 CANDIDATES')
print('=' * 120)

for idx, row in promising.head(10).iterrows():
    print()
    print('=' * 60)
    print(f" {row['symbol']} | Score: {row['score']:.2f}")
    print('=' * 60)

    # Technical Indicators
    print("  Technical Indicators:")
    adx_1h = row.get('adx_1h', 0)
    adx_15m = row.get('adx_15m', 0)
    adx_5m = row.get('adx_5m', 0)
    print(f"    ADX (1h/15m/5m): {adx_1h:.1f} / {adx_15m:.1f} / {adx_5m:.1f}")
    print(f"    RSI (15m): {row.get('rsi_15m', 0):.1f}")
    print(f"    EMA Slope (1h): {row.get('ema_slope_1h', 0):.4f}")
    print(f"    BB Width: {row.get('bb_width', 0):.4f}")
    print(f"    Range Size: {row.get('range_size_pct', 0):.2f}%")

    # Grid Parameters
    print("  Grid Parameters:")
    print(f"    Number of Grids: {int(row.get('num_grids', 0))}")
    print(f"    Profit per Grid: {row.get('profit_per_grid_pct', 0):.2f}%")

    # HMM Regime
    print("  HMM Regime Detection:")
    print(f"    Range Probability: {row.get('hmm_range_prob', 0):.4f}")
    print(f"    Trend Probability: {row.get('hmm_trend_prob', 0):.4f}")

    # Stochastic Features
    print("  Stochastic Features:")
    print(f"    Survival Probability: {row.get('survival_prob', 0):.3f}")
    print(f"    Hurst Exponent: {row.get('hurst_exponent', 0):.3f}")
    ou = row.get('ou_halflife')
    print(f"    OU Halflife: {ou:.1f} bars" if pd.notna(ou) else "    OU Halflife: -")

    # Market Data
    print("  Market Data:")
    print(f"    Last Price: ${row.get('last_price', 0):.4f}")
    vol = row.get('quote_volume_24h', 0)
    print(f"    24h Volume: ${vol/1e6:.2f}M")
    print(f"    Funding Rate: {row.get('funding_rate', 0)*100:.4f}%")
    oi = row.get('open_interest', 0)
    print(f"    Open Interest: ${oi/1e6:.2f}M")

    # Notes
    notes = row.get('notes', '')
    if notes:
        print(f"  Notes: {notes[:80]}...")

print()
print('=' * 120)
print('Results saved to: results/neutralgrid_candidates_20260120_160852.csv')
print('=' * 120)
