# QRT Challenge — Accuracy Improvement Roadmap

## Scope and evidence used

This is an analysis-only audit of `qrt_challenge_pipeline_final.ipynb`. The notebook was not modified, no model was executed, and no predictions were generated. The public leaderboard score is treated only as an external transport check, never as a tuning target.

The current production baseline is:

- experiment: `P2_catboost_binary_purged`;
- model: CatBoost binary classification;
- features: `benchmark_plus_cross_and_confidence`;
- encoding: smoothed allocation positive-rate encoding;
- preprocessing: median imputation plus standard scaling (`linear_standard`);
- validation: four purged expanding folds, each with 252 validation timestamps and a 20-timestamp embargo;
- decision threshold: `0.492`;
- ensemble: rejected;
- public leaderboard accuracy: `0.5097`.

The saved notebook results establish the following reference points:

| Result | Mean/global accuracy |
|---|---:|
| Official 8-fold LightGBM benchmark | 52.219% |
| E10 feature/encoding configuration | 52.466% |
| CatBoost binary, shuffled 8-fold timestamp CV | 52.631% |
| CatBoost binary, purged expanding CV | 52.378% mean; 52.374% pooled OOF |
| Purged CatBoost fold range | 51.928%–52.885% |
| Public leaderboard | 50.970% |

Additional train-only descriptive checks, which do not fit a model, show:

- 17.88% of rows have `|target| <= 0.0005`, and 34.44% have `|target| <= 0.001`;
- the positive-class rate across ten chronological training blocks ranges from 49.43% to 52.13%;
- `SIGNED_VOLUME_1` is missing on 387,506 of 527,073 rows (73.52%); its missing rate ranges from 71.62% to 76.31% across the ten time blocks;
- the observed positive rate is 50.33% when `SIGNED_VOLUME_1` is missing and 51.79% when it is present. This is descriptive and may be explained by time or allocation composition, but it makes missingness worth testing;
- each allocation belongs to exactly one group. Group sample sizes are large (100,170 to 168,913 rows), although group-within-timestamp counts still need a minimum-size guard for robust cross-sectional statistics.

There is one reproducibility caveat in the saved notebook state. The threshold-calibration cell is saved with execution disabled, the final-diagnostics cell consequently ends in `ValueError: Calibrer P2_catboost_binary_purged before the diagnostics`, and the ensemble output says it was skipped. The requested audit supplies the validated conclusions (threshold `0.492`, informative confidence, rejected ensemble), which are accepted here, but the granular fold thresholds, final error tables, and model-error correlations are not present in the saved outputs. Recovering and persisting those existing OOF diagnostics should precede new training.

## 1. Executive summary

### What currently works

The pipeline is methodologically strong. Complete timestamps are kept together, model comparison uses a fixed feature pipeline, target-derived allocation encoding is learned inside outer training folds, the final candidates are checked with past-to-future purged validation, threshold selection is nested rather than fitted on the evaluated fold, and an unsuccessful ensemble was rejected. Binary classification is also empirically the right default: CatBoost classification beat CatBoost regression by about 0.22 percentage points under the same standard-CV pipeline, and the same direction holds for LightGBM and XGBoost.

The strongest demonstrated information is not a particular boosting library. Under purged validation CatBoost leads LightGBM by only 0.018 percentage points and XGBoost by 0.026 percentage points. The useful evidence is instead the combination of recent-return behavior, global cross-sectional context, and allocation history.

### What likely limits performance

1. **Deployment mismatch and non-stationarity.** The public result is 1.41 percentage points below the purged mean and 0.96 percentage points below the worst purged fold. The target balance and key missingness rate move over time. Equal weighting of all historical timestamps may therefore be suboptimal.
2. **CatBoost has not been tested in its natural numeric regime.** `linear_standard` median-imputes all missing values before CatBoost. Scaling is not needed by a split-based tree, and median imputation prevents CatBoost from separating missing values natively. This is especially important for `SIGNED_VOLUME_1`.
3. **The allocation encoding is static and not strictly chronological inside the training fold.** It is safe with respect to the outer validation fold, but its internal cross-fitting shuffles timestamps. A past-only, hierarchical, optionally decayed encoding would better match the final train-to-future-test use case.
4. **The existing cross-sectional block is global, not group-relative.** It already contains timestamp percentiles, ordinary z-scores, and mean deviations for eight bases. Recreating those would add no new information. Robust median/MAD measures and `TS × GROUP` ranks are genuinely absent.
5. **Research degrees of freedom are becoming a risk.** Many choices have already been compared on the same folds. Small gains can be meta-overfit even when each individual experiment is leakage-safe.

### Is `0.5097` consistent with the validation evidence?

It is plausible, but not comfortably predicted by the reported validation distribution. It is below every purged fold, so ordinary fold-to-fold sampling variation alone is not a convincing explanation. Temporal drift, the public subset composition, repeated selection on the same folds, and the mismatch between shuffled internal target encoding and future deployment can plausibly account for the gap. The score is a warning about transportability, not evidence that the pipeline necessarily leaks, and it must not be used to choose the next model.

### Highest-priority next experiments

1. **CatBoost native missing values plus a compact missingness block.** Test no scaling/no imputation, then add only explicit availability features and a `SIGNED_VOLUME_1` missing flag.
2. **Recency weighting versus a fixed recent window.** Determine directly whether older timestamps help or hurt on fixed purged future folds.
3. **Strictly past-only hierarchical allocation encoding.** Replace shuffled internal target encoding with expanding allocation → group → global shrinkage; test one decayed variant.
4. **Compact group-relative robust cross-sectional features.** Add only `TS × GROUP` ranks, median deviations, robust z-scores, and group breadth/dispersion.
5. **A small temporal-summary block, followed only later by compact CatBoost tuning.** Add interpretable EWM momentum/volatility and short-versus-long features before spending degrees of freedom on model parameters.

No immediate production code or model change is justified. First recover the existing OOF calibration/error diagnostics, freeze the validation protocol, and run the experiments above one family at a time.

## 2. Current pipeline strengths

### Correct unit of validation

All rows from a timestamp remain in the same fold. This is essential for a panel problem because row-wise splitting would leak contemporaneous cross-sectional information and underestimate uncertainty. Fold-integrity checks verify row separation, complete timestamp assignment, temporal order, and embargo separation.

### Reproducible baseline and controlled ablation

The official benchmark is reproduced before extensions are introduced. E0–E10 mostly change one hypothesis at a time, and the best combined configuration improves the official benchmark by about 0.25 percentage points. Rejected blocks—return-volume, timestamp regime, and turnover—were not silently retained.

### Appropriate objective choice

The challenge scores every row by binary sign accuracy. Direct binary objectives were better than regression followed by a sign threshold for all three boosted-tree libraries in the controlled comparison. Regression remains a useful diversity reference, but it should not replace binary classification without new evidence.

### Leakage-aware preprocessing

Numeric preprocessing and category mappings are fitted within each outer training fold. The allocation target encoding does not use outer-validation targets, and unseen categories have a defined fallback. Feature builders reject target columns and preserve row alignment.

### Temporal robustness check

The final model is not selected solely from shuffled timestamp folds. CatBoost, LightGBM, and XGBoost were all re-evaluated under the same purged expanding design. Their performance falls modestly but remains above the naive and official references.

### Conservative thresholding and ensemble decision

The selected threshold is derived from OOF probabilities rather than leaderboard feedback. The simple blend of three similar boosted-tree models was rejected rather than kept for sophistication alone. This is the correct decision when diversity does not convert into nested OOF gain.

## 3. Current pipeline weaknesses / open questions

### Model limitations

- The CatBoost configuration is a single reasonable default-like point: 500 trees, depth 5, learning rate 0.03, and Logloss. Regularization and capacity have not been tested in a controlled compact search.
- CatBoost receives median-imputed, standardized values. Standardization adds no expected information to tree splits; median imputation can destroy a predictive missingness state.
- `GROUP` and `ALLOCATION` are not passed as native CatBoost categorical variables in the final model, even though a raw-category configuration exists in the notebook. The model receives only a numeric allocation target encoding.
- The three ensemble members use the same 93 processed features, folds, target formulation, and broadly similar tree learners. Their small score differences and failed blend suggest highly overlapping errors. Actual error correlation must be persisted before any further ensemble work.
- Class weights are not justified: the target is already close to balanced. Asymmetric losses would optimize a problem different from the leaderboard metric.

### Feature limitations

- The final 93 columns comprise 53 benchmark features, 24 global timestamp-relative features, 15 RET1-confidence features, and one allocation encoding. The raw 20 return and 20 volume lags may contain redundant or noisy columns, but no stable family-level pruning has been completed.
- Existing cross-sectional features already include percentiles, ordinary z-scores, and deviations from timestamp means for `RET_1`, return means over 3/5 lags, volatility over 5/20 lags, `SIGNED_VOLUME_1`, five-lag mean signed volume, and turnover. New work must not duplicate these.
- There are no robust median/MAD cross-sectional positions, group-relative ranks, allocation-versus-group deviations, or group breadth/dispersion features.
- There are no compact exponentially weighted return/volatility summaries, short-minus-long momentum, momentum acceleration, or sign-persistence summaries.
- Existing missingness information is weakly represented. The rejected return-volume block included volume missing counts, and the rejected turnover block included a turnover-missing flag, but those indicators were bundled with many unrelated interactions. A small missingness-only block is materially different.
- `SIGNED_VOLUME_1` is extremely sparse while other recent volume lags are much more complete. Blind median imputation makes “unobserved” indistinguishable from “typical.”

### Validation uncertainty

- The official eight-fold scheme shuffles timestamps. It is useful for benchmark comparability, not as the final deployment estimate.
- The purged system validates four 252-timestamp blocks, covering about 40% of training rows OOF. It is directionally appropriate, but only four fold-level observations are available.
- The 20-timestamp purge appears linked to the 20 input lags, but the notebook does not establish the target horizon or execute an overlap audit that proves 20 is the correct independence gap. A purge protects against label-horizon overlap; shared lagged covariates are a dependence issue, not automatically label leakage.
- The experiment summary's `improved_fold_count` is zero throughout because candidate results do not carry aligned benchmark fold scores into that calculation. It should not be used as evidence of cross-fold consistency.
- Repeatedly choosing features, models, thresholds, and ensembles on the same four purged folds creates meta-overfitting. From now on, the first three purged folds should be development folds and the latest fold should be a one-time confirmation for each shortlisted family.

### Potential non-stationarity

- Positive-class prevalence varies by roughly 2.7 percentage points across ten chronological training blocks.
- `SIGNED_VOLUME_1` availability also changes over time.
- Shuffled-CV CatBoost falls from 52.631% to 52.378% under purged future validation, and the public result falls further. These observations do not prove that recency weighting will help, but they clearly justify a small, pre-specified weighting/window experiment.
- Older data may reduce variance while introducing stale relationships. Only paired performance on identical future validation blocks can decide this tradeoff.

### Calibration issues

- The `0.492` threshold reportedly gives a small nested gain, so its uncertainty matters. Fold-specific selected thresholds and the near-optimal interval are not preserved in the saved outputs.
- The current nested selector learns each fold's threshold from all other folds, including folds later in time. This is valid ordinary nested CV, but not a strict simulation of a threshold deployed into the future. A chronological threshold audit should use only earlier OOF blocks.
- The high test predicted-positive rate is a sanity observation, not a label-based tuning signal. It cannot justify moving the threshold.
- Group- or regime-specific thresholds add several degrees of freedom and should be attempted only if stable calibration offsets repeat across development folds. Any such thresholds must be shrunk toward the global threshold.

## 4. Ranked experiment roadmap

`Runtime` is relative to the saved purged CatBoost run of about 169 seconds for four folds on the current machine. It excludes implementation time.

| Priority | Experiment | Hypothesis | Expected upside | Runtime | Implementation complexity | Leakage risk | Overfitting risk | Validation required |
|---|---|---|---|---|---|---|---|---|
| P0 | Recover OOF diagnostics and freeze the protocol | Current OOF predictions can reveal whether errors, thresholds, and model disagreements are stable before new training | No direct gain; prevents false discoveries | Very low; no fitting | Low | None if only existing OOF is used | Low | Persist fold thresholds, paired errors, confidence/magnitude/missingness tables, and timestamp-block paired intervals |
| P0 | Native-NaN CatBoost and compact missingness block | Median imputation suppresses a materially different and potentially predictive missing state | Small to modest, plausibly 0.05–0.25 pp OOF | 3–4 purged CatBoost runs | Low | Low | Low–medium | Fixed purged dev folds, one locked confirmation, paired timestamp bootstrap |
| P0 | Recency decay and rolling-window training | Older observations contain stale relationships that dilute recent signal | Small to medium if drift is real; otherwise negative | 3 purged CatBoost runs | Low | Low | Medium | Same future blocks and features; compare equal weight, two decays, and one window without changing threshold |
| P0 | Past-only hierarchical allocation encoding | Allocation history is useful, but chronological shrinkage and recency better match deployment | Small to modest, plausibly 0.05–0.25 pp | 3–4 purged runs | Medium–high | Low if emitted before same-TS updates | Medium | Unit tests for same-TS isolation; fixed purged folds; one confirmation |
| P1 | Group-relative robust cross-sectional block | Four groups have different behavior; within-group relative position contains information absent from global ranks | Small, plausibly 0.05–0.20 pp | 2 purged runs | Medium | Low | Medium | Minimum group×timestamp count guard; leave-block-in/out on fixed folds |
| P1 | Compact EWM momentum/volatility block | Fixed arithmetic means miss decay, acceleration, and sign persistence | Small, plausibly 0.03–0.20 pp | 2 purged runs | Low–medium | Low | Medium | One core block, then at most one extension; no arbitrary indicator library |
| P1 | Family-level feature pruning | Weak raw volume/lag families increase variance and hurt transport | Usually small, 0–0.15 pp, with possible runtime reduction | Low for permutation; 3–5 runs for ablation | Medium | Low | Medium | OOF family permutation followed by pre-specified leave-one-family-out and locked confirmation |
| P1 | Compact CatBoost search (10 configurations maximum) | Current depth/regularization may be slightly mismatched to the weak signal | Small, 0–0.20 pp | About 8–10 dev-fold runs plus one confirmation | Low | Low | Medium–high | Tune only after data representation is fixed; same dev folds; one winning config reaches confirmation |
| P1 | Mild target-magnitude sample weighting | Near-zero future returns are sign-ambiguous and may consume model capacity | Small; can easily be negative because all rows count equally | 2 purged runs | Low | Low | Medium–high | Weights derived only from training targets; unweighted OOF accuracy; report magnitude strata |
| P2 | Native `GROUP`/`ALLOCATION` CatBoost categories with time order | Ordered categorical statistics and group pooling may capture interactions missed by one numeric encoding | Uncertain, potentially small | 2–3 purged runs | Medium | Low at the outer fold; chronology must be explicit | Medium–high | Compare raw group only, raw allocation+group, and best past-only encoding; never use global target statistics |
| P2 | Purge/window sensitivity audit | The chosen embargo or expanding window may not match dependence and target horizon | Improves reliability, not necessarily accuracy | 3 baseline validations | Low | Low | Low if diagnostic only | Pre-specify purge 0/20/40 or use the known target horizon when documented; do not select by leaderboard |
| P2 | Chronological/global and shrunk group thresholds | Stable class-prior/calibration offsets may justify a threshold other than 0.5 | Very small | OOF-only after a final model exists | Medium | Low if nested past-only | High | Earlier-fold-only selection; group thresholds require repeated direction and shrinkage to global |
| P2 | Structurally diverse ensemble | A time-window or magnitude-weighted model may correct different rows than the baseline | Small and conditional | OOF-only blend after candidates exist | Medium | Low | High | Gate on standalone accuracy, disagreement, error correlation, and nested paired gain |
| P3 | Ranking objectives | Within-timestamp ranking is not the scored objective and still needs an unstable sign threshold | Low | Medium–high | Medium | Low | High | Not recommended without a prior diagnostic showing relative-order errors dominate |
| P3 | Separate group models or global-plus-group residual models | Specialization may help a difficult group | Low given only four groups and loss of pooled data | 4× model cost | High | Low | High | Consider only if group interactions fail and fold-stable residual structure remains |
| P3 | Asymmetric loss or class weights | A class imbalance might justify unequal error costs | Near zero; target is balanced and metric is symmetric | Low | Low | Low | Medium | Not recommended under current evidence |

## 5. Detailed design of the top experiments

### Common research protocol

Use this protocol for every modeling experiment below:

1. Keep the current four purged expanding validation blocks and their row assignments fixed. Use folds 0–2 for development and do not inspect the candidate's fold-3 result until one configuration from that experiment family is selected.
2. Compare all candidates with threshold `0.5` first. Threshold calibration is model-specific and should not hide whether the representation or training objective improved.
3. Use pooled OOF accuracy on identical validation rows as the primary metric. Also record every fold delta, predicted-positive rate, log loss as a probability-quality diagnostic, and the paired difference in correctness.
4. Resample complete timestamps to estimate a confidence interval for the paired accuracy delta; never bootstrap rows independently.
5. A candidate reaches confirmation only if its development pooled gain is at least 0.08 percentage points, it is non-negative in at least two of three development folds, and no development fold loses more than 0.15 percentage points.
6. Keep a candidate only if the four-fold pooled gain is at least 0.10 percentage points, at least three of four fold deltas are positive, the confirmation fold does not reverse the hypothesis, and the gain is not concentrated in a handful of timestamps. These thresholds are deliberately conservative because many experiments have already used related folds.
7. After a modeling candidate is kept, run chronological nested threshold calibration once. Do not use threshold performance to rescue a model that loses at `0.5`.

The official shuffled eight-fold score can be reported afterward for continuity, but it is not the selection metric.

### Experiment 0 — OOF diagnostic and validation freeze

**Limitation addressed.** The notebook's saved state does not contain the detailed final diagnostics, calibration fold table, or model-error correlations needed to target failure modes.

**Variables/output to create.** Using the already generated purged OOF probabilities only:

- correctness and confidence by fold, chronological block, group, allocation support, turnover quintile, volatility quintile, momentum quintile, target-magnitude quintile, and `SIGNED_VOLUME_1` availability;
- selected threshold and near-optimal interval by fold;
- a stricter walk-forward threshold table in which fold `k` uses only earlier validation blocks;
- pairwise prediction disagreement, both-wrong rate, one-model-only-correct rates, and error correlation for CatBoost, LightGBM, and XGBoost;
- paired timestamp-block bootstrap intervals for CatBoost versus the other two models;
- a manifest containing row IDs, fold IDs, model parameters, feature columns, and hashes of the input files.

**Training configuration.** None. Do not refit any estimator.

**Comparison baseline.** The stored `P2_catboost_binary_purged` OOF output at 0.5 and at 0.492.

**Acceptance criterion.** This is an evidence gate, not a model. It is complete when the outputs are persisted and reproduce the saved aggregate accuracies exactly.

**Stopping criterion.** Stop if the required OOF arrays cannot be recovered; do not rerun models merely to make charts until the later experiment protocol is frozen.

**Cost and risk.** Very low compute, no leakage risk. The main risk is post-hoc storytelling; therefore only use the tables to choose among the pre-ranked families above, not to invent dozens of subgroups.

### Experiment 1 — Native missing values and sparse-feature handling

**Limitation addressed.** CatBoost currently receives standardized, median-imputed numerics. This removes the distinction between missing and median values, even though `SIGNED_VOLUME_1` is missing on 73.52% of rows and has a descriptive class-rate difference.

**Candidates.** Keep the model, folds, feature families, allocation encoding, and random seed fixed.

1. `N1_native`: imputer `none`, scaler `none`, unchanged 93 logical inputs.
2. `N2_native_missing_flags`: `N1` plus:
   - `MISS_SIGNED_VOLUME_1`;
   - `MISS_TURNOVER`;
   - `VOLUME_AVAILABLE_COUNT_5` and `VOLUME_AVAILABLE_COUNT_20`;
   - `VOLUME_LEADING_MISSING_STREAK` from lag 1 outward;
   - `RET_AVAILABLE_COUNT_20`.
3. `N3_indicator_without_sparse_raw`: `N2` but drop raw `SIGNED_VOLUME_1` and its three existing global cross-sectional transformations. This tests whether the sparse value adds information beyond its availability state and older volumes.

Do not add missingness-by-target statistics. If historical missing rates by allocation are later tested, fit them on the outer training fold and update them past-only.

**Training configuration.** Current CatBoost binary parameters: 500 iterations, depth 5, learning rate 0.03, Logloss, seed 42. CatBoost officially supports numeric NaNs and can consider a split separating missing from observed values; scaling is therefore unnecessary for this test ([CatBoost missing-value processing](https://catboost.ai/docs/en/concepts/algorithm-missing-values-processing), [common parameters](https://catboost.ai/docs/en/references/training-parameters/common)).

**Comparison baseline.** Current `linear_standard` CatBoost at threshold 0.5.

**Acceptance and stopping.** Apply the common gate. Stop the family if `N1` does not at least match the baseline and neither missingness candidate reaches the development threshold. Keep only one representation.

**Cost and risks.** Three CatBoost CV candidates; low leakage risk and low-to-medium overfitting risk. The main confounder is that missingness may proxy time or allocation rather than a stable mechanism, which is why fold and time-block deltas are required.

### Experiment 2 — Recency weighting and fixed recent windows

**Limitation addressed.** Equal-weight expanding training assumes stationarity. The chronological class balance, missingness rate, validation gap, and leaderboard gap make that assumption uncertain.

**Candidates.** For each outer fold, define `age` as the number of training timestamps between a row and the last training timestamp. All weights are calculated from training timestamps only and normalized to mean one.

1. `T0_equal`: current expanding baseline.
2. `T1_decay_slow`: `w = 2^(-age / (0.50 * n_train_ts))`.
3. `T2_decay_fast`: `w = 2^(-age / (0.25 * n_train_ts))`.
4. `T3_recent_half`: train on only the most recent 50% of available training timestamps, preserving the same embargo and validation block.

Do not tune a continuous half-life. These three alternatives answer a single question: whether stale data hurt enough to offset their variance reduction.

**Training configuration.** Use the winning preprocessing from Experiment 1 and otherwise unchanged CatBoost. Pass decay weights only to training rows. Target encoding must use the same weighting rule if it remains target-derived; otherwise the model and encoding would represent different histories.

**Comparison baseline.** `T0_equal` evaluated on exactly the same validation rows.

**Additional validation.** Report deltas as a function of fold chronology and training-set length. If decay helps only the earliest fold but hurts later folds, the non-stationarity hypothesis is not supported for deployment.

**Acceptance and stopping.** Apply the common gate. Select at most one of `T1`–`T3`. Stop if both decays and the recent window lose on development; that is direct evidence that older observations still help.

**Cost and risks.** Three new CatBoost candidates. No leakage if weights use only timestamp order. Overfitting risk is medium because half-life/window choice is a model parameter; pre-specification and the locked fold control it.

### Experiment 3 — Past-only hierarchical allocation history

**Limitation addressed.** Allocation target encoding gave one of the largest validated feature gains, but the current encoder uses shuffled timestamp cross-fitting for training-row values and a single smoothing strength of 20. It does not adapt to drift or share strength explicitly through group.

**Safe construction.** Sort the outer training fold by timestamp. For every timestamp:

1. emit encodings using statistics through the preceding timestamp only;
2. use no targets from the current timestamp while emitting any row at that timestamp;
3. after all rows at the timestamp are encoded, update allocation, group, and global counts;
4. for outer validation rows, freeze statistics at the last outer-training timestamp. Do not update from embargo or validation labels.

Create:

- `ALLOC_HIST_RATE`: allocation positive rate shrunk to its group prior, then to the global prior;
- `ALLOC_HIST_LOG_COUNT`: `log1p` of prior allocation observations, so the model can learn encoding reliability;
- optionally `ALLOC_SIGN_HIST_RATE`: allocation × `sign(RET_1)` rate, backed off to group × sign, then allocation, group, and global priors.

**Candidates.** Limit the family to:

1. expanding hierarchy with allocation/group prior strengths 20/100;
2. stronger shrinkage 100/500;
3. the better of the first two plus the conditional-sign encoding;
4. the better expanding hierarchy with exponentially decayed counts, fixed half-life 504 timestamps.

Do not concatenate all candidates. Replace the current allocation target encoding during the first comparison so the source of any gain is identifiable.

**Training configuration.** Winning preprocessing from Experiment 1, equal-weight baseline unless Experiment 2 has already passed confirmation, and unchanged CatBoost parameters.

**Comparison baseline.** Current fold-safe `allocation_target` encoding with smoothing 20.

**Acceptance and stopping.** Apply the common gate and require unit tests proving that changing targets at timestamp `t` cannot change encodings at `t` or earlier. Stop after one smoothing comparison if neither version improves development OOF.

**Cost and risks.** Three to four model runs plus non-trivial feature implementation. Leakage risk is low only with emit-before-update semantics. Overfitting risk is medium because conditional cells are smaller; hierarchical fallback and count features are mandatory.

### Experiment 4 — Group-relative robust cross-sectional context

**Limitation addressed.** The current block measures only global timestamp-relative position with means and standard deviations. The four groups are large and exhibit different descriptive performance, but group is nested in allocation and not otherwise exposed to the final model.

**Features.** For the four bases `RET_1`, `mean(RET_1..3)`, `mean(RET_1..5)`, and `std(RET_1..20)`, calculate within each `TS × GROUP` cell:

- percentile rank;
- value minus group median;
- robust z-score `(value - median) / (1.4826 * MAD + epsilon)`.

Add only three group context features:

- share of positive `RET_1` within `TS × GROUP`;
- group MAD of `RET_1`;
- group median short-minus-long momentum, where short is lags 1–3 and long is lags 6–20.

Add one relative-momentum feature: group percentile of short momentum minus group percentile of long momentum. This is 16 features, not a broad technical-indicator library.

For `TS × GROUP` cells with fewer than 20 observed values, set robust statistics to missing and optionally provide the cell count. Do not use targets.

**Training configuration.** Add this block to the winning prior pipeline; unchanged CatBoost parameters.

**Comparison baseline.** Same pipeline without the new block. A second candidate may add raw `GROUP` as a CatBoost categorical feature, but only after the numeric group-relative block is evaluated.

**Acceptance and stopping.** Apply the common gate. Require gains in at least two groups and no group loss worse than 0.20 percentage points; this prevents an overall gain driven only by the largest group. Stop after the compact block and at most one raw-group extension.

**Cost and risks.** Two model runs, low leakage risk because all features use contemporaneously available predictors only, and medium overfitting risk due to correlated ranks.

### Experiment 5 — Compact temporal summaries over the 20 observed lags

**Limitation addressed.** Raw lags and arithmetic means do not encode smooth decay, momentum acceleration, or sign persistence explicitly.

**Core features.** Treat `RET_1` as the most recent observed return and use only the row's 20 known lags:

- exponentially weighted mean return with half-lives 2, 5, and 10 lags;
- exponentially weighted volatility with half-lives 5 and 10 lags;
- short-minus-long momentum: mean lags 1–3 minus mean lags 6–20;
- acceleration: mean lags 1–3 minus mean lags 4–6;
- positive-sign share over lags 1–5 and 1–10;
- lag-1 autocorrelation across the observed return sequence, with a minimum valid-pair count.

Only if the core block passes development may one small extension add maximum drawdown over the lagged cumulative-return path and volatility-of-volatility. Turnover trend is not available because turnover has no lag sequence. Same-lag return-volume correlations and missing counts were already tested in the rejected return-volume block and should not be recreated.

**Training configuration.** Winning pipeline to date, unchanged CatBoost.

**Comparison baseline.** Same pipeline without the temporal block.

**Acceptance and stopping.** Apply the common gate. Stop after the core block if it fails; do not search alternative windows.

**Cost and risks.** One or two model runs, low leakage risk, medium overfitting risk from correlated summaries.

### Experiment 6 — Compact CatBoost parameter search

**Limitation addressed.** The current CatBoost setting may be slightly too shallow, too weakly regularized, or trained for an inefficient iteration/learning-rate budget. Tuning before fixing missingness and encodings would conflate representation and capacity, so this experiment comes later.

**Search.** Fix features, preprocessing, weights, encoding, folds, and seed. Evaluate no more than ten configurations on development folds:

1. depth 4, 5, 6 with 500 iterations and learning rate 0.03;
2. depth 4, 5, 6 with 800 iterations and learning rate 0.02;
3. clone the best schedule with `l2_leaf_reg=10`;
4. clone it with `random_strength=0.5`;
5. clone it with `random_strength=2.0`;
6. clone it with `border_count=128`.

The current configuration is included, so only nine additional points are created. `Logloss` remains the optimized loss; report Accuracy as an evaluation metric. Do not add class weights. Do not use the outer validation fold for early stopping; a fixed iteration budget preserves an honest comparison. CatBoost documents depth, L2 regularization, random split strength, bootstrap settings, and numeric border count as distinct controls ([CatBoost training parameters](https://catboost.ai/docs/en/references/training-parameters/)).

**Comparison baseline.** The fully selected data pipeline with current 500/depth-5/0.03 parameters.

**Acceptance and stopping.** Rank on the three development folds, send only one configuration to confirmation, and apply the common keep rule. If the top two development configurations differ by less than 0.03 percentage points, prefer the shallower or more regularized one.

**Cost and risks.** About ten three-fold fits plus one confirmation. Leakage risk is low; meta-overfitting risk is the highest among the recommended experiments, which is why the search is capped and sequenced late.

### Experiment 7 — Mild target-magnitude weighting

**Limitation addressed.** Plain Logloss gives the same training importance to a barely positive return and a large positive return. Near-zero signs may be economically ambiguous or measurement-sensitive, and more than one third of labels have absolute magnitude at most 0.001.

**Candidates.** Compute magnitude cutoffs separately inside every outer training fold. Validation remains completely unweighted.

1. `W1_bottom_decile_half`: weight 0.5 for the lowest training-fold decile of `|target|`, otherwise 1.0.
2. `W2_smooth`: `weight = clip(0.5 + |target| / median_train(|target|), 0.5, 2.0)`, normalized to mean one.

Do not delete near-zero rows. The official metric weights them equally, so hard exclusion creates a sharper train/evaluation mismatch. Do not select cutoffs from validation targets.

**Training configuration.** Final selected pipeline before threshold calibration, CatBoost binary Logloss, training weights only.

**Comparison baseline.** Equal-weight CatBoost on the same rows and folds.

**Additional validation.** Report unweighted overall accuracy and accuracy by target-magnitude quintile. The hypothesis is supported only if gains on clearer labels more than compensate for any loss on near-zero rows in the official unweighted metric.

**Acceptance and stopping.** Apply the common gate. Stop after these two pre-specified schemes. Regression and CatBoost regression have already underperformed and should not be repeated unchanged. Ordinal magnitude classes are P2 at best and should be considered only if mild weighting shows a stable gain.

**Cost and risks.** Two model runs, low leakage risk, medium-to-high overfitting risk because weights use the training label and do not match the evaluation weights.

### Experiment 8 — Family-level feature selection

**Limitation addressed.** Ninety-three processed features are not large in absolute terms, but weak lag/volume families can still increase tree variance when the signal is only a few percentage points above chance.

**Stage 1: diagnostic importance.** On already fitted development-fold models, calculate the paired accuracy loss after permuting one family at a time in its validation fold. For allocation-specific numeric features, permute within timestamp where feasible so the market regime is preserved. Families are:

- raw returns;
- raw signed volumes;
- turnover;
- benchmark return summaries;
- benchmark timestamp aggregates;
- cross-sectional block;
- RET1-confidence block;
- allocation history;
- any newly accepted block.

CatBoost split importance and SHAP may explain models, but neither is sufficient for selection. Stability of the validation permutation effect across folds is the main screen.

**Stage 2: retrained ablation.** Retrain at most three leave-one-family-out candidates: the families with near-zero or negative stable permutation effect. Include a specific `drop SIGNED_VOLUME_1 only` candidate only if Experiment 1 did not already resolve it.

**Comparison baseline.** The winning full feature set.

**Acceptance and stopping.** Apply the common gate. A smaller model may be kept when accuracy is statistically indistinguishable and runtime or stability improves, but it must not be claimed as an accuracy gain without positive paired evidence.

**Cost and risks.** OOF permutation is cheap; up to three model runs follow. Leakage risk is low. Overfitting risk is medium because feature selection reuses validation results, so confirmation is mandatory.

### Experiment 9 — Threshold stability after final model selection

**Limitation addressed.** The reported improvement from `0.492` is slight, fold stability is not saved, and the current nested scheme is not strictly chronological.

**Procedure.** After all model choices are frozen:

1. evaluate the global nested selector as currently implemented;
2. evaluate a walk-forward selector where each validation block's threshold is estimated only from earlier OOF blocks; use 0.5 for the first block or estimate it inside the preceding training period;
3. record the threshold curve in `[0.48, 0.52]`, its near-optimal plateau, predicted-positive rate, and paired accuracy delta versus 0.5;
4. compare the raw optimum with a 50% shrinkage toward 0.5: `0.5 + 0.5*(t_hat - 0.5)`.

Group-specific thresholds are allowed only as a secondary test if all development folds show the same directional group calibration offset. Estimate them nested, impose a minimum group support, and shrink them strongly toward the global threshold. Do not create timestamp-regime thresholds from the test distribution.

**Training configuration.** None; OOF-only after final model training.

**Comparison baseline.** Threshold 0.5, plus the current global 0.492 as a reported reference.

**Acceptance and stopping.** Keep a non-default threshold only if the chronological nested gain is positive, the optimum lies on a reasonably broad plateau, fold thresholds do not alternate materially around 0.5, and the paired gain is not driven by one time block. Otherwise use 0.5 or the shrunk threshold. Never submit multiple tiny threshold variants.

**Cost and risks.** Negligible compute, no target leakage when nested correctly, but high selection-overfitting risk because threshold effects are small.

## 6. Experiments NOT recommended

- **Unchanged regression followed by sign thresholding.** CatBoost, LightGBM, and XGBoost regression have already lost to their binary counterparts under the same pipeline.
- **A large ordinal or custom-loss project now.** Mild magnitude weighting is a simpler test of the near-zero-label hypothesis while preserving the binary submission objective.
- **Hard removal of all near-zero targets.** Those rows still count equally on the leaderboard. At most, test two mild training-weight schemes.
- **Global timestamp ranks, ordinary z-scores, or mean deviations for the existing eight bases.** They are already in the final cross-sectional block.
- **The original return-volume, timestamp-regime, or turnover blocks unchanged.** They were already rejected. A small missingness-only block or a genuinely lagged lead/lag hypothesis is different and must be labeled as such.
- **Hundreds of technical indicators or arbitrary window searches.** With only four purged folds, their research degrees of freedom would overwhelm any small gain.
- **Whole-training or whole-dataset target statistics.** Allocation, group, conditional, and regime encodings must be outer-fold fitted and, for the proposed improvement, past-only with no same-timestamp target updates.
- **Test-set-driven feature, model, hyperparameter, or threshold selection.** Test predicted-positive rate and leaderboard results are sanity checks only.
- **Leaderboard threshold tuning or repeated submissions around 0.492.** One final submission should follow a frozen OOF decision.
- **A huge hyperparameter optimizer.** Ten targeted CatBoost configurations are enough; Bayesian or random search would meta-overfit the same folds.
- **Class weights or asymmetric losses.** The target is close to balanced and both classes have equal scoring cost.
- **LightGBM/XGBoost ranking objectives at this stage.** Ranking within timestamp optimizes relative order, not independent sign accuracy, and still needs a threshold. It adds complexity without a diagnosed ranking failure.
- **Another blend of the current three probabilities.** It has already failed. Revisit an ensemble only after a time-window, target-formulation, or categorical model creates demonstrably different OOF errors.
- **Separate models for each group now.** Group sample sizes are large, but separate models discard useful pooling and multiply tuning decisions. Test group-relative features or a raw group category first.
- **Feature selection from CatBoost impurity importance or SHAP alone.** Use them for interpretation; use fold-stable OOF permutation and retrained family ablations for decisions.
- **Selecting a purge because it scores best.** The purge should follow the documented target horizon/dependence structure. Sensitivity analysis is a robustness check, not another tuning axis.

## 7. Suggested next sequence

```text
Recover and persist existing OOF diagnostics
        |
        v
Freeze folds 0–2 for development and fold 3 for confirmation
        |
        v
Experiment 1: native missing / missingness representation
        |
        +-- no stable gain --> retain current preprocessing
        |
        +-- stable gain ----> adopt one representation
        |
        v
Experiment 2: equal history vs two decays vs recent-half window
        |
        +-- no stable gain --> retain equal expanding history
        |
        +-- stable gain ----> adopt one weighting/window rule
        |
        v
Experiment 3: past-only hierarchical allocation encoding
        |
        v
Experiment 4 and 5, one block at a time:
group-relative context -> compact temporal summaries
        |
        v
Family-level pruning on the best representation
        |
        v
At most 10 CatBoost parameter configurations
        |
        v
Optional two-scheme magnitude weighting
        |
        v
One locked purged confirmation of the final candidate
        |
        v
Chronological nested threshold calibration and stability check
        |
        v
Train once on all training data -> generate one new submission
```

Do not combine two unconfirmed feature families. If an experiment fails its development gate, stop that family and move to the next hypothesis. If several small changes pass independently, add them sequentially and verify that their gains survive together; individual deltas are not assumed additive.

An ensemble is considered only after the final single-model candidates exist. A second model must (a) be within 0.10 percentage points of the best standalone development accuracy, (b) show materially different pairwise errors rather than merely different probabilities, and (c) improve the locked nested blend. Otherwise retain the single CatBoost model.

## 8. Estimated potential

### Likely small gains

Native missing handling, one compact missingness block, group-relative features, temporal summaries, pruning, and modest CatBoost regularization are each more likely to yield hundredths to a few tenths of a percentage point than a large jump. Several may be redundant, so gains should not be added arithmetically.

### Medium-risk gains

Recency weighting, rolling windows, past-only decayed allocation history, and magnitude weighting can produce a larger improvement if the hidden period differs materially from older training data. They can also reduce accuracy by discarding useful history or mismatching the equal-weight metric. Their upside comes with higher regime and selection risk.

### Unlikely large gains

A sustained gain above roughly 0.5 percentage points in honest future validation would probably require a genuine structural discovery—such as a strong chronological allocation effect or a previously hidden group-relative mechanism—not routine tuning. A much larger claimed gain from many features or thresholds should be treated first as a leakage, split, or meta-overfitting warning.

The realistic objective is therefore not to promise a leaderboard score. It is to improve transportability, accept only repeatable paired OOF gains, and make at most one new submission after the full research decision is frozen.
