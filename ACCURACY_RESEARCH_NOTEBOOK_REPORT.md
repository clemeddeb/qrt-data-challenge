# Accuracy Research Notebook Report

## Deliverable

- Research notebook: `qrt_challenge_accuracy_research.ipynb`
- Purpose: controlled, leakage-safe evaluation of the accuracy-improvement roadmap against the unique validated CatBoost baseline.
- Production notebook: `qrt_challenge_pipeline_final.ipynb` was not modified.

The research notebook is valid notebook JSON (`nbformat` 4.5), contains no execution outputs, and has all code-cell execution counts set to `None`.

## Notebook structure

The notebook contains 16 numbered sections:

1. Research objective and current baseline
2. Setup and data loading
3. Frozen validation protocol
4. Recover the current baseline
5. Common evaluation and acceptance rules
6. Experiment A — Native missing values
7. Experiment B — Recency weighting
8. Experiment C — Past-only hierarchical allocation encoding
9. Experiment D — Group-relative robust cross-sectional features
10. Experiment E — Compact temporal summaries
11. Experiment F — Feature-family pruning
12. Experiment G — Compact CatBoost tuning
13. Experiment H — Optional target-magnitude weighting
14. Candidate selection and locked confirmation
15. Final chronological threshold calibration
16. Final research summary

Cell counts:

- Markdown cells: **17**
- Code cells: **16**
- Total cells: **33**

Each main code cell or experiment block is preceded by a short explanation covering **What**, **How**, **Why**, and **Connection to the project**.

## Experiment families included

### A — Native missing values

- A0: exact validated baseline
- A1: native NaNs without scaling or imputation
- A2: native NaNs plus five compact missingness/availability features
- A3: optional removal of sparse raw `SIGNED_VOLUME_1` while retaining availability information

### B — Recency weighting

- B0: equal-weight control, identified as equivalent to A0
- B1: slow exponential decay
- B2: faster exponential decay
- B3: most recent 50% of training timestamps

Allocation target statistics follow the same time-weight convention for B1–B3.

### C — Past-only hierarchical encoding

- C1: allocation → group → global history with 20/100 smoothing
- C2: stronger 100/500 shrinkage
- C3: fixed 504-timestamp decayed history

The encoder emits features before updating a timestamp, so no row uses same-timestamp or future labels.

### D — Group-relative cross-sectional context

One compact 16-feature block contains ranks, median deviations, MAD z-scores, positive RET1 breadth, dispersion, group momentum, and relative-momentum rank spread. `TS × GROUP` cells require at least 20 observations.

### E — Temporal summaries

Ten row-local features cover EWM return means, EWM volatility, short-minus-long momentum, acceleration, sign persistence, and lag-1 autocorrelation.

### F — Feature-family pruning

- OOF family permutation within timestamp where possible
- At most three manually selected leave-one-family-out retraining tests
- Requires a previously confirmed representation
- SHAP is not used as a selection rule

### G — CatBoost tuning

- Six depth/iteration/learning-rate stage-one configurations
- Four optional stage-two regularization/quantization variants around one human-selected stage-one result
- Ten configurations maximum
- No random search, Bayesian optimization, or confirmation-fold early stopping

### H — Magnitude weighting

- H1: half weight for the smallest training-fold magnitude decile
- H2: smooth clipped magnitude weights
- No rows are deleted and validation accuracy remains unweighted
- Allocation encoding remains equal-weight so this experiment changes the training loss only

## Frozen validation design

The notebook reconstructs the production purged expanding protocol:

- four expanding folds;
- 20-timestamp embargo;
- 10% validation window;
- folds 0, 1, and 2 are development folds;
- fold 3 is the locked confirmation fold.

Development helpers call `get_development_folds()` and raise `PermissionError` if fold 3 is requested. Every candidate uses the same validation timestamps as the baseline.

The conservative development gates are declared as constants before any experiment runs:

- pooled improvement at least `+0.0008`;
- non-negative delta in at least two of three development folds;
- no development fold below `-0.0015` versus baseline.

The final confirmation gates require:

- positive locked-fold gain;
- four-fold pooled improvement at least `+0.0010`;
- positive deltas in at least three of four folds;
- timestamp-block bootstrap lower bound no worse than `-0.0002`.

These are documented as conservative research rules, not universal statistical laws.

## Confirmation-fold protection

`SELECTED_CONFIRMATION_EXPERIMENT_ID` defaults to `None`, and `RUN_CONFIRMATION` defaults to `False`.

The confirmation helper checks that:

1. an experiment ID was explicitly selected;
2. it matches the requested confirmation run;
3. its cached development comparison passed the development gate;
4. it is not the A0 baseline;
5. only fold 3 is evaluated.

No family automatically receives confirmation results. Feature pruning, tuning, and magnitude weighting additionally require a representation already stored in `CONFIRMED_EXPERIMENT_CONFIGS`.

## Central research objects

- Baseline contract: `RESEARCH_BASELINE_CONFIG`
- Results cache: `RESEARCH_RESULTS_CACHE`
- Central experiment table: `RESEARCH_EXPERIMENT_LOG`
- Confirmed configurations: `CONFIRMED_EXPERIMENT_CONFIGS`
- Frozen representation selector: `FROZEN_REPRESENTATION_EXPERIMENT_ID`
- Explicit confirmation selector: `SELECTED_CONFIRMATION_EXPERIMENT_ID`
- Final compact hand-off: `FINAL_RESEARCH_SUMMARY`

The common evaluation engine reports pooled OOF accuracy, log loss, predicted-positive rate, fold metrics, paired correctness differences, and timestamp-block bootstrap intervals.

## Heavy execution flags

All flags are defined, referenced by their experiment block, and default to `False`:

- `RUN_NATIVE_MISSING_EXPERIMENTS = False`
- `RUN_RECENCY_EXPERIMENTS = False`
- `RUN_HISTORICAL_ENCODING_EXPERIMENTS = False`
- `RUN_GROUP_RELATIVE_EXPERIMENTS = False`
- `RUN_TEMPORAL_FEATURE_EXPERIMENTS = False`
- `RUN_FEATURE_PRUNING = False`
- `RUN_CATBOOST_TUNING = False`
- `RUN_MAGNITUDE_WEIGHTING = False`
- `RUN_CONFIRMATION = False`
- `RUN_FINAL_THRESHOLD_CALIBRATION = False`

No estimator was fitted and no predictions were produced during notebook creation.

## Static and synthetic tests performed

The following checks passed:

- notebook parses as valid JSON;
- all 16 code cells compile;
- no duplicate top-level function names exist;
- all ten heavy flags are present, used, and false;
- all code cells are unexecuted and have empty outputs;
- the research baseline contract matches the final notebook configuration;
- development helper access to fold 3 raises `PermissionError`;
- past-only encoder: changing labels at a timestamp cannot change encodings emitted at that timestamp;
- past-only encoder: changing future labels cannot change earlier encodings;
- past-only encoder: an unseen allocation receives a finite hierarchical fallback;
- feature-building contracts assert preservation of index and `ROW_ID`;
- target columns are removed before feature construction and rejected if they enter the model frame;
- the test file is read only with `nrows=0` for schema validation;
- `X_test_schema` is not referenced by development, confirmation, tuning, calibration, or selection helpers;
- no model-training cell was executed.

The historical-encoding invariance tests were run against the notebook's extracted pure chronological core, without importing or running a model library.

## Baseline notebook integrity

The production notebook SHA-256 remained:

`dffe1546b3aaf15b8e0341786e1feaf2501de38a88365b9f575ed0707663cb90`

This matches the hash recorded before research-notebook creation. The production notebook was therefore unchanged.

## Intended final action

The research notebook contains no submission generation. Its hand-off instruction is:

> If the final candidate passes confirmation, transfer its configuration to the production notebook and make one new submission.
