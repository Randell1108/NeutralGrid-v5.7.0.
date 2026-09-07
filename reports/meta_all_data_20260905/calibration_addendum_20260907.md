# Calibration diagnosis — September 7, 2026

The earlier comparison reported three sigmoid-calibrated models. It varied feature counts and estimators, but did not select among calibration methods. This follow-up recalculates metrics from the already saved raw and calibrated probabilities; no models were refitted and no holdout labels were changed.

The separate calibration partition contained 867 rows with a 57.79% FASTWIN-positive rate. The independent latest holdout contained 490 rows with a 41.02% positive rate. All three calibrated models predicted approximately 54% on average. The measured failure is predominantly overprediction on this later cohort. The difference in population rates and outcome-start conventions is evidence of a calibration mismatch; it does not uniquely identify a market-regime cause.

| Model | Raw ECE | Sigmoid-calibrated ECE | Raw mean prediction | Calibrated mean prediction |
|---|---:|---:|---:|---:|
| Selected 24 features | 0.09766 | 0.15593 | 48.60% | 54.16% |
| Baseline 20 features | 0.10861 | 0.14073 | 50.64% | 54.34% |
| Prior-feature reference, 19 features | 0.11064 | 0.13782 | 50.66% | 54.29% |

The raw 24-feature probabilities therefore already fall below 0.10 on this particular holdout. Their AUC remains 0.69991 and Brier score improves from 0.23349 to 0.21695 compared with the sigmoid output. This is a post-hoc diagnostic discovery, not a newly established promotion pass. The margin below 0.10 is small, there are only four held-out scan groups, and choosing the raw variant after seeing these outcomes consumes the holdout for that choice. The research-source and production-profile restrictions also remain in force.

For the selected model, the third held-out scan had an actual FASTWIN rate of 26.53% against an average prediction of 52.10%; the fourth had 35.35% against 56.90%. Its first scan was not overpredicted in aggregate, so a uniform retrospective subtraction is not an established fix. ECE measures weighted gaps within probability bins and is not identical to the overall mean-prediction bias.

The next valid calibration experiment should:

1. Use consistent candidate-availability timing and the same full seven-hour FASTWIN outcome contract for model fitting, calibration and testing. Historical scan-anchored labels and post-availability labels must not be silently treated as interchangeable.
2. Compare raw probabilities, sigmoid calibration, and isotonic calibration where sufficient independent calibration data exist. Compare predeclared recent calibration windows in chronological development folds. Fit the classifier and calibrator on separate past partitions, retaining the existing purge gaps. Preserve the ten-bin ECE calculation and 0.10 threshold; also assess Brier score, AUC and uncertainty.
3. Freeze the entire classifier/calibration choice before evaluating multiple new, complete scan groups. The September 6 cohort can contribute only after its full conservative replay window completes, September 7 at approximately 04:51 America/Lima. Its 111 candidates constitute one scan group and do not alone establish broad temporal robustness.

Scikit-learn documents disjoint fitting/calibration data and warns that isotonic calibration is more prone to overfitting with small datasets; approximately 1,000 calibration samples is a guideline, not a guarantee. See the [official calibration documentation](https://scikit-learn.org/stable/modules/calibration.html). The existing 867-row calibration partition alone is a reason to treat isotonic cautiously.

## Previous champion records

The saved August 23 UTC promotion decision reports:

| Historical record | Samples | Stored OOF ECE | Recorded status |
|---|---:|---:|---|
| Incumbent champion `20260822_230654` | 1,386 | 0.02288 | pass |
| Replacement candidate in that decision | 5,849 | 0.08151 | pass |

Neither reported an ECE exceeding 0.10. These are stored historical out-of-fold results, not measurements on the new September holdout. The decision explicitly labels the champion/candidate comparison `incomparable_champion_absolute_gate_only` because their evaluation-contract metadata differ. The current native evaluator also has the previously documented shuffled-fold/unpurged-fallback limitation; that is not evidence that the old champion's actual forward ECE exceeded 0.10. Its September forward ECE has not been established here.

The August 27 recovery shadow separately reports OOF ECE 0.03533, but it is a diagnostic artifact and should not be conflated with the previous production champion. The 20-feature reference retrained in this study is likewise not a replay of the old champion's exact weights.

Evidence: `calibration_diagnosis_20260907.json`, `holdout_predictions.csv`, `models/meta_labeler_promotion_decision.json`, and the recovery shadow's `metadata.json`. No production artifact was changed and no temporary files were created by this follow-up.
