# Training and validation

Run `python -m src.training.pipeline --seed 42` from the repository root in
`phishing-ml`. Verify completion with `python -m src.training.pipeline --verify`.
Neither command opens test or PhishTank data. Existing feature transformers and
splits are read only; URL logistic regression fits its own StandardScaler on
training inside its saved Pipeline.

Email: logistic regression (liblinear, max_iter=1000) and LinearSVC (dual=auto,
max_iter=10000), each with no weighting and balanced weighting. URL: scaled
logistic regression and 100-tree random forest with min_samples_leaf=3 and two
workers. Native numerical libraries are limited to two threads. Sparse email
features remain sparse.

Rank configurations using validation average precision, default-threshold
phishing F1, then recall. For the selected model, maximize phishing F1 across
validation precision-recall thresholds; ties prefer recall. LinearSVC thresholds
are uncalibrated margins. Other models use positive-class probabilities. For
inference, apply the existing feature transformer, load selected_model.joblib,
and compare the positive score against selected_threshold.json using >=.
Selected models are symlinks to their configuration artifact, avoiding duplicate
model bytes. Only load trusted local joblib files.

Comparison reports include default-threshold metrics, ranking metrics, confusion
matrices in [0,1] order, training/scoring time, convergence warnings, selected
threshold metrics, input file metadata, and inherited label/data limitations.
Reports are under reports/training; model artifacts are under models/email and
models/url. training_summary.json marks completed verification of both tasks.

Each task writes a start marker before fitting. A failed/partial run is never
automatically retrained; inspect /tmp/phishing-training.log before deciding how
to recover. Successful tasks are verified and skipped on subsequent invocations.
