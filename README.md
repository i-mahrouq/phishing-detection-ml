# Phishing detection — Email V1 and URL V1

Two independent offline research classifiers: **email** uses word TF-IDF plus six numerical indicators with a linear SVM; **URLs** use 13 lexical features with a Random Forest. Predictions are model outputs, not safety guarantees. The interface is the CLI.

## Setup

Use Python **3.11** and run commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively, run `conda env create -f environment.yml` followed by `conda activate phishing-ml`. No GPU is required.

## Run pretrained models

From repository root with Python 3.11 and requirements installed:

```bash
python -m src.inference.predict email --subject 'Meeting' --body 'Project meeting tomorrow at ten.' --json
python -m src.inference.predict url --url 'https://www.example.com' --json
python -m src.inference.predict --interactive
```

No datasets, retraining, network calls or other task’s artifacts are needed for a prediction. Keep the selected model, its threshold and schema together. Email also requires the saved feature transformer. Preserve the existing `selected_model.joblib` relative symlinks and load only trusted joblib files.

Predictions use `score >= saved threshold`. Email scores are uncalibrated SVM margins; URL scores are Random Forest class estimates. Neither is a guaranteed real-world risk percentage.

## Verified V1 results

| Task | Test rows | Accuracy | Phishing precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Email / linear SVM | 44,947 | 0.996618 | 0.984090 | 0.983673 | 0.983881 |
| URL / Random Forest | 35,202 | 0.995995 | 0.999058 | 0.991519 | 0.995274 |

PhishTank positive-only external recall is 0.986638 (24,440/24,771 detected). It cannot measure false positives because it contains no legitimate examples. Original results are preserved in [reports/evaluation](reports/evaluation).

Email label provenance and URL source-format bias limit generalization. For example, URL V1 flags `https://example.com/` as phishing while accepting `https://www.example.com`. High internal scores do not establish production readiness.

## Layout and checks

```text
src/        preprocessing, features, training, evaluation and CLI inference
models/     selected V1 artifacts, candidate models and feature schemas
data/       local raw_data, processed, splits and features
reports/    frozen training comparisons and evaluation evidence
tests/      maintained preprocessing, training, evaluation and inference checks
docs/       local academic guides and technical references (ignored by Git)
archive/    local historical research, notebooks and audit tools (ignored by Git)
```

Explicitly named V2/V3 artifact directories remain at their original paths to preserve historical references. V2 experiments completed; V3 stopped at source qualification, without a final model. Neither replaces the V1 classifiers used by the CLI.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
python -m src.preprocessing.urls --verify
python -m src.features.pipeline --verify
python -m src.training.pipeline --verify
python -m src.evaluation.pipeline --verify
```

The maintained suite has 22 tests, including independent Email-only and URL-only artifact bundles. Saved-run verifiers require the matching local datasets, feature manifests and candidate models. A code-only clone can run pretrained inference and the maintained tests; it does not contain the full experimental data.

## Reproduce from raw data

Use a fresh output root containing the exact original datasets under `data/raw_data/`: Enron maildir, `Nazario_5.csv`, `email_text.csv`, the PhiUSIIL CSV and the PhishTank verified-online CSV. Sources are discovered locally; preprocessing does not download them. Current upstream snapshots may differ from the historical inputs, and the exact provenance of the two email CSVs remains unresolved.

```bash
REPRO_ROOT=/absolute/path/to/fresh-reproduction
python -m src.preprocessing.emails --root "$REPRO_ROOT" --seed 42 --workers 4
python -m src.preprocessing.urls --root "$REPRO_ROOT" --seed 42
python -m src.features.pipeline --root "$REPRO_ROOT" --seed 42
python -m src.training.pipeline --root "$REPRO_ROOT" --seed 42
python -m src.evaluation.pipeline --root "$REPRO_ROOT"
```

Both preprocessing tasks must complete before the shared feature stage. Email preprocessing replaces exports, so preserve the existing run and reproduce in a separate root. Models and reports from the new run stay under that root. Original processed data in this checkout is at `data/processed/`.

## Local academic documentation

The `docs/` directory is excluded from Git and is not supplied by a fresh clone. When available locally, read these guides in order:

1. `docs/ACADEMIC_REPORT.md` — objectives, methodology, results and limitations.
2. `docs/EXECUTION_GUIDE.md` — setup, exact input paths, reproduction and troubleshooting.
3. `docs/DEFENSE_MASTER_GUIDE.md` — study priorities, examination questions and presentation rehearsal.

Supporting technical explanations, cleanup records and verification evidence are retained under `docs/reference/`. Documentation, datasets and archived research remain on disk; ignoring them does not delete their contents.
