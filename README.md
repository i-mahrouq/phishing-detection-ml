# Phishing Detection with Machine Learning

An academic cybersecurity project with two independent classifiers:

- **Email V1** classifies email content using word TF-IDF, six numerical indicators and a linear Support Vector Machine (SVM).
- **URL V1** classifies HTTP(S) addresses using 13 lexical features and a Random Forest.

Both use the project labels **0 = legitimate** and **1 = phishing**. The classifiers provide predictions, not guarantees that a message or destination is safe. Email CSV label provenance and URL source-format bias limit real-world generalization.

## 1. What the project does

| Component | Input | Processing | Selected model |
|---|---|---|---|
| Email | Optional subject and a required decoded plain-text or HTML body | Clean text, count links before HTML cleaning, apply frozen word TF-IDF and six scaled indicators | Unweighted `LinearSVC` |
| URL | One URL beginning with `http://` or `https://` | Validate syntax and calculate 13 ordered lexical features from the original string | `RandomForestClassifier` |

The pipelines share utility code but have separate models, features and thresholds. Email analysis does not automatically run the URL classifier on extracted links. Inference does not train models, read datasets, visit URLs or perform DNS lookups. Attachments, sender authentication and destination webpage content are outside its scope.

## 2. Installation

Use **Python 3.11**. Run every command below from the repository root. Linux with Python 3.11.18 was verified; other operating systems have not been verified. No GPU is required.

Choose one environment setup:

**Python virtual environment**

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Conda**

```bash
conda env create -f environment.yml
conda activate phishing-ml
```

Check the active environment:

```bash
python --version
python -m pip check
```

Runtime and reproduction dependencies are pinned in `requirements.txt`. To run tests, also install the development requirements:

```bash
python -m pip install -r requirements-dev.txt
```

## 3. Using the interface

The current interface is a **terminal application (CLI)**. The previously developed Streamlit interface was removed; this version has no web-server launch command.

### Interactive menu

```bash
python -m src.inference.predict --interactive
```

The application prompts for `email`, `url` or `quit`:

1. Enter `email` to provide a subject and body. The subject may be blank. Finish a multiline body with a single `.` on its own line.
2. Enter `url` to provide one complete HTTP(S) address.
3. Enter `quit` to exit. Short forms `e`, `u` and `q` are also accepted.

Example email input:

```text
Choose email / url / quit: email
Subject: Meeting
Body (finish with a single . on its own line):
Project meeting tomorrow at ten.
.
```

This example produces a legitimate prediction with an SVM margin of approximately `-1.194796`. The program prints the model score, decision threshold, heuristic risk band and explanatory notes.

### Analyze an email directly

```bash
python -m src.inference.predict email \
  --subject 'Meeting' \
  --body 'Project meeting tomorrow at ten.'
```

The body must contain visible text; a subject alone is insufficient. Supply decoded message content, not an entire raw MIME envelope. The CLI cannot reconstruct MIME alternatives, headers or attachments that were not supplied. Combined subject/body input is limited to 5,000,000 characters.

### Analyze a URL directly

```bash
python -m src.inference.predict url --url 'https://www.example.com'
```

Supply an HTTP(S) URL with a valid hostname/IP and port, without embedded whitespace. URLs are limited to 65,536 characters. Syntax validation does not establish destination safety.

### JSON output and model location

Add `--json` for machine-readable output:

```bash
python -m src.inference.predict email --body 'Project meeting tomorrow' --json
python -m src.inference.predict url --url 'https://www.example.com' --json
```

To use a separate artifact root, place global `--root` before the subcommand:

```bash
python -m src.inference.predict --root /absolute/path/to/artifact-root \
  url --url 'https://www.example.com' --json
```

For all options:

```bash
python -m src.inference.predict --help
python -m src.inference.predict email --help
python -m src.inference.predict url --help
```

### Understanding the result

| Field | Meaning |
|---|---|
| `prediction` | `PHISHING` or `LEGITIMATE`, using `score >= threshold` |
| `score` | Native SVM margin for email, or Random Forest class-1 estimate for URLs |
| `threshold` | Saved validation-selected decision threshold |
| `model` | Selected candidate name |
| `score_type` | `uncalibrated_margin` or `probability` |
| `risk_level` | Provisional LOW/MEDIUM/HIGH band based on distance from the threshold |
| `score_note`, `risk_note` | Interpretation limits for the score and band |

Email uses threshold `-0.24541511286701478`; URL uses `0.4754530853344011`. An SVM margin is not a probability. A Random Forest estimate is not calibrated real-world risk. Risk bands use threshold ±0.5 for email and ±0.10 for URLs; they do not change the classification.

**Known limitation:** URL V1 flags `https://example.com/` as phishing while accepting `https://www.example.com`. This illustrates sensitivity to formatting and source bias; it is not evidence that example.com is malicious. Do not change the frozen threshold based on a demonstration input.

Inputs are not persistently logged by application code, but command arguments may be visible in shell history or process listings. Interactive mode avoids placing the body in command arguments. Load only trusted local joblib files.

## 4. Repository structure

```text
phishing-detection-ml/
├── README.md
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── environment.yml
├── pytest.ini
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── emails.py
│   │   └── urls.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── common.py
│   │   ├── email_features.py
│   │   ├── url_features.py
│   │   └── pipeline.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── pipeline.py
│   └── inference/
│       ├── __init__.py
│       └── predict.py
├── tests/                     # Seven maintained test modules listed below
├── models/
│   ├── email/
│   ├── url/
│   ├── feature_engineering/
│   └── url_v2/                # Historical; not used by V1 inference
├── data/                      # Local, ignored by Git
│   ├── raw_data/
│   ├── processed/
│   ├── splits/
│   └── features/
├── reports/
│   ├── training/              # V1 validation/model selection
│   ├── evaluation/            # V1 test/external evidence
│   ├── url_v2/                # Historical experiments
│   └── url_v3/                # Historical source qualification
├── docs/                      # Local, ignored by Git
│   ├── ACADEMIC_REPORT.md
│   ├── EXECUTION_GUIDE.md
│   ├── DEFENSE_MASTER_GUIDE.md
│   └── reference/
└── archive/                   # Local, ignored by Git
    ├── research/
    ├── notebooks/
    ├── tools/
    └── audit_history/
```

### Root files

| File | Purpose |
|---|---|
| `README.md` | Project introduction, setup, usage, file map and reproduction commands |
| `.gitignore` | Keeps local datasets, documentation, archives, caches and secrets out of new Git additions; allowlists selected models and aggregate reports |
| `requirements.txt` | Pinned dependencies for V1 inference, preprocessing, training and evaluation |
| `requirements-dev.txt` | Runtime requirements plus pytest |
| `environment.yml` | Alternative Conda setup specifying Python 3.11 |
| `pytest.ini` | Maintained test discovery and repository import-path configuration |

### Source files

| File | Responsibility |
|---|---|
| `src/preprocessing/emails.py` | Discover MIME/CSV sources, decode messages, clean text, extract links, group duplicates, exclude conflicts and export grouped splits |
| `src/preprocessing/urls.py` | Validate/normalize URLs, create canonical duplicate keys, map labels, split by hostname and exclude internal overlap from external data |
| `src/features/common.py` | Shared suspicious keyword list and counting function |
| `src/features/email_features.py` | `EmailFeatures`: train-fitted word TF-IDF plus six MaxAbs-scaled numerical indicators; reused unchanged at inference |
| `src/features/url_features.py` | `URLFeatures`: stateless extraction of the fixed 13 lexical columns |
| `src/features/pipeline.py` | Generate and verify saved matrices, row alignment, schemas and feature reports for both tasks |
| `src/training/pipeline.py` | Train four email and two URL configurations; select on validation AP, tune the selected threshold on validation F1, save and verify artifacts; expose shared score extraction |
| `src/evaluation/pipeline.py` | Evaluate frozen test/external features, calculate metrics and overlaps, and save confusion plots/error indices |
| `src/inference/predict.py` | CLI/menu, input validation, lazy artifact loading, schema checks, feature transformation and threshold-based prediction |
| Each package `__init__.py` | Retains the package/import structure |

Do not casually rename `src.features.email_features.EmailFeatures`: joblib loads the saved transformer through that class path. Inference imports the shared `scores` helper from the training module, but importing it does not start training.

### Models and schemas

| Location / files | Role |
|---|---|
| `models/email/linear_svm_none.joblib` | Selected email classifier |
| `models/email/linear_svm_balanced.joblib`, `logistic_regression_none.joblib`, `logistic_regression_balanced.joblib` | Preserved email comparison candidates; also checked by training verification |
| `models/url/random_forest.joblib` | Selected URL classifier |
| `models/url/logistic_regression.joblib` | Preserved scaled logistic-regression baseline |
| `models/email/selected_model.joblib`, `models/url/selected_model.joblib` | Relative symlinks to the selected candidate in the same directory |
| `models/email/selected_threshold.json`, `models/url/selected_threshold.json` | Selected model reference, threshold, score type, label/rule and validation metrics |
| `models/email/training_started.json`, `models/url/training_started.json` | Preserved start markers and partial-run evidence |
| `models/feature_engineering/email_transformer.joblib` | Frozen email vocabulary, IDF weights and numerical scaler |
| `models/feature_engineering/emails_schema.json`, `urls_schema.json` | Frozen feature layout/type contracts |
| `models/url_v2/`, `models/feature_engineering/urls_v2/` | Historical candidate models, character vectorizer and configuration; not V1 runtime inputs |

For **Email-only inference**, keep the selected email model/alias/threshold, email transformer and email schema. For **URL-only inference**, keep the selected URL model/alias/threshold and URL schema. Neither requires the other task's artifacts, candidate models, reports or datasets. Keep the source package layout shown above.

### Datasets and generated files

| Location / file family | Role |
|---|---|
| `data/raw_data/emails/enron/maildir/` | Original MIME messages, presumed legitimate by source membership |
| `data/raw_data/emails/nazario/Nazario_5.csv`, `email_text.csv` | Mixed-label email sources; exact lineage and phishing-versus-spam semantics remain uncertain |
| `data/raw_data/urls/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv` | Internal URL source; original label 1 is inverted to project 0, and original 0 to project 1 |
| `data/raw_data/urls/phishtank/online-valid.csv` | Provider-verified online phishing snapshot for positive-only external assessment |
| `data/processed/emails/emails_cleaned.parquet` | Cleaned/deduplicated messages with labels and provenance |
| `data/processed/emails/work.sqlite` | Preserved parsing/grouping workspace; contains email content |
| `data/processed/urls/urls_cleaned.parquet`, `phishtank_external.parquet` | Cleaned internal URLs and overlap-excluded external positives |
| `data/processed/{emails,urls}/cleaning_report.json` | Source counts, exclusions, label rules, split counts and limitations |
| `data/splits/{emails,urls}/{train,validation,test}.parquet` | Authoritative frozen partition membership and provenance |
| `data/features/emails/{train,validation,test}.npz` | Sparse 50,006-column float32 email feature matrices |
| `data/features/emails/{train,validation,test}_rows.parquet` | Matching row identities and labels |
| `data/features/urls/{train,validation,test,external}.parquet` | 13 lexical predictors plus identity/target metadata |
| `data/features/{emails,urls}/feature_report.json` | Input metadata, schemas and alignment/completion checks |
| Explicit `urls_v2`/`urls_v3` directories, OpenPhish and Tranco sources | Historical external experiments/source qualification, not extra V1 training sources |

Generated datasets and raw originals remain local. ZIP and notebook-checkpoint copies are preserved provenance, not additional training examples. Record-level corpus files share these schemas; they are not separate software modules.

### Evaluation evidence

| Files under `reports/` | Purpose |
|---|---|
| `training/{email,url}_model_comparison.json` | Candidate validation metrics, fit times, selected model/threshold and input metadata |
| `training/training_summary.json` | Completion summary for both tasks |
| `evaluation/{email,url}_test_metrics.json` | Frozen held-out test results, confusion counts and source/overlap analysis |
| `evaluation/phishtank_external_metrics.json` | External phishing detections, misses and recall |
| `evaluation/{email,url}_confusion_matrix.png` | Saved V1 test matrix visualizations |
| `evaluation/{email,url}_test_errors.parquet` | Error row indices, labels, scores and types; no raw email/URL content |
| `evaluation/evaluation_started.json`, `evaluation_summary.json` | Preserved evaluation run state and frozen-artifact verification |
| `url_v2/`, `url_v3/` | Historical training/selection/diagnostic and source-qualification evidence |

### Tests

| File | Checks |
|---|---|
| `tests/test_email_preprocessing.py` | MIME/HTML parsing, duplicates/groups, conflicts, large inputs and exports |
| `tests/test_url_preprocessing.py` | URL normalization, labels, overlap removal and completed runs |
| `tests/test_features.py` | Training-only fitting, stable feature order, finiteness and row alignment |
| `tests/test_training.py` | Tiny synthetic fits, train/validation boundaries, selection and serialization |
| `tests/test_evaluation.py` | Metric formulas, single-class restrictions, error artifacts and model compatibility |
| `tests/test_inference.py` | Frozen preprocessing/scoring parity, validation, JSON and interactive behavior |
| `tests/test_packaging.py` | Offline prediction and independent Email-only/URL-only artifact bundles without datasets |

### Local documentation and archive

`docs/` is ignored by Git and is not included in a fresh clone. When available locally, start with:

1. `docs/ACADEMIC_REPORT.md` — academic context, methodology, results and limitations.
2. `docs/EXECUTION_GUIDE.md` — detailed execution, inputs and troubleshooting.
3. `docs/DEFENSE_MASTER_GUIDE.md` — study order, technical questions and presentation rehearsal.

`docs/reference/` contains architecture, datasets/source details, both pipelines, algorithms, evaluation, codebase map, the defense question bank, cleanup report and audit evidence.

`archive/research/` holds original V2/V3 source/configuration/tests/environment snapshots; `archive/notebooks/` holds exploratory notebooks; `archive/tools/` holds optional audit utilities; `archive/audit_history/` holds earlier records. These local materials are not prerequisites for V1 inference. Historical artifact/data paths were retained to avoid rewriting evidence or adding compatibility links.

**Version boundary:** V2 completed experiments, including a character TF-IDF SVM, but showed severe failure on Tranco candidate negatives. V3 completed dataset/source qualification only; it has no final trained model. Archived imports require separate historical-layout reconstruction and are not part of maintained test discovery.

## 5. Run checks

With development dependencies installed:

```bash
python -m pytest -q
python -m pip check
```

The last cleanup verification passed all **22 maintained tests**, including both independent artifact bundles. Tests use tiny temporary training fixtures where needed; they do not retrain the preserved project models.

For the complete saved local run:

```bash
python -m src.preprocessing.urls --verify
python -m src.features.pipeline --verify
python -m src.training.pipeline --verify
python -m src.evaluation.pipeline --verify
```

These verifiers require the matching local data, candidate artifacts and reports. A code-only clone can run pretrained inference and maintained tests, but cannot verify or regenerate the original experiment without those additional files.

## 6. Reproduce preprocessing, training and evaluation

Run reproduction in a **fresh output root** to preserve existing evidence. Place the exact original raw snapshots under that root using this layout:

```text
<reproduction-root>/data/raw_data/
├── emails/
│   ├── enron/maildir/<employee>/<folder>/<message>
│   └── nazario/
│       ├── Nazario_5.csv
│       └── email_text.csv
└── urls/
    ├── phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv
    └── phishtank/online-valid.csv
```

V1 discovers local sources; it does not download them. New provider feeds may differ from the historical snapshots. Exact independent reproduction is also limited by unresolved email CSV provenance and path/mtime-sensitive manifests.

From the code repository root:

```bash
REPRO_ROOT=/absolute/path/to/fresh-reproduction
python -m src.preprocessing.emails --root "$REPRO_ROOT" --seed 42 --workers 4
python -m src.preprocessing.urls --root "$REPRO_ROOT" --seed 42
python -m src.features.pipeline --root "$REPRO_ROOT" --seed 42
python -m src.training.pipeline --root "$REPRO_ROOT" --seed 42
python -m src.evaluation.pipeline --root "$REPRO_ROOT"
```

Both preprocessing tasks must complete before the shared feature stage. Email preparation groups related messages; URL preparation separates exact hostnames. Both target approximately 70/15/15 train/validation/test partitions. Learned transforms fit training only. Candidates are selected on validation AP, then the selected threshold maximizes validation F1; held-out evaluation follows selection.

Email preprocessing replaces its working tables/exports, so do not rerun it against the preserved run merely to resolve a missing-file error. Completed later stages verify/skip; partial-run markers require inspection. Do not edit manifests or remove markers to force success.

Verify and use the newly generated artifacts:

```bash
python -m src.features.pipeline --root "$REPRO_ROOT" --verify
python -m src.training.pipeline --root "$REPRO_ROOT" --verify
python -m src.evaluation.pipeline --root "$REPRO_ROOT" --verify
python -m src.inference.predict --root "$REPRO_ROOT" url --url 'https://www.example.com' --json
```

Models and reports stay under the new root. A full original-corpus retraining was not performed merely for cleanup.

## 7. Preserved V1 results and limitations

| Task | Test rows | Accuracy | Phishing precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Email / linear SVM | 44,947 | 0.996618 | 0.984090 | 0.983673 | 0.983881 |
| URL / Random Forest | 35,202 | 0.995995 | 0.999058 | 0.991519 | 0.995274 |

Email test false positives/negatives are **75/77**; URL test false positives/negatives are **14/127**. PhishTank external recall is **0.986638**: 24,440 detections and 331 misses out of 24,771 labeled positives. That positive-only set cannot measure false positives. Original evidence is preserved in [reports/evaluation](reports/evaluation) and [reports/training](reports/training).

These scores were independently recomputed during the audit without changing the models. They describe historical snapshots. Source bias, uncertain email labels, related domain families and absent contemporary representative external negatives prevent claims of production readiness. No probability calibration or saved training-performance study is supplied.

## 8. Troubleshooting

| Problem | Action |
|---|---|
| Missing `python` or imports | Activate the Python 3.11 environment and install requirements with that interpreter |
| Model/config missing | Check the named artifact and selected-model symlink target; restore the matching trusted model, schema and threshold |
| joblib class/version error | Keep the source module layout and pinned dependency versions |
| Invalid URL | Include an HTTP(S) scheme and valid hostname/IP/port; remove embedded whitespace |
| Empty email | Provide visible body text; the subject alone is insufficient |
| Manifest/frozen-input mismatch | Inspect the original files, root and timestamps; do not bypass checks or regenerate preserved splits automatically |
| Dataset missing in a clone | Obtain the exact authorized local sources; inference itself does not require them |
| Unexpected confident prediction | Review source/format limitations; a high score is not proof of maliciousness |
