# Phishing Detection — Email and URL

Two independent offline machine-learning components:

| Task | Algorithms compared | Current selected model | Features |
|---|---|---|---|
| Email | Logistic Regression and Linear SVM, each with original weight variants | Linear SVM | Word TF-IDF + six scaled numerical indicators |
| URL | Scaled Logistic Regression and Random Forest | Random Forest | 13 lexical URL indicators |

Project labels are **0 legitimate, 1 phishing**. Predictions are research outputs, not safety guarantees. Source/label uncertainty and formatting bias limit generalization.

## Setup

Use Python **3.11**, from repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Alternatively, `conda env create -f environment.yml` then `conda activate phishing-ml`. Install `requirements-dev.txt` for pytest. No GPU, external web server, network lookup or dataset is needed for pretrained prediction. Only load trusted local joblib artifacts.

## Local web interface

After setup, run from the repository root:

```bash
python -m src.web.app
```

Open **http://127.0.0.1:8000**. Use the Email form for subject/body or the URL form for an HTTP(S) address. Each form calls its existing inference module and shows the prediction, model score and saved threshold. Models load on the first request and remain cached until restart.

Stop with `Ctrl+C`. If port 8000 is busy, run `python -m src.web.app --port 8001` and open http://127.0.0.1:8001. The server listens only on this computer; it needs no extra dependencies or build step. Inputs are not saved and URLs are not visited.

## Email commands

```bash
python -m src.inference.email --subject 'Meeting' --body 'Project meeting tomorrow at ten.' --json
python -m src.preprocessing.emails --verify
python -m src.features.email_pipeline --verify
python -m src.training.train_email --verify
python -m src.training.train_email --compare
python -m src.evaluation.evaluate_email --verify
```

Inference accepts an optional subject and required decoded visible body, with a 5,000,000-character combined limit. HTML links are counted before cleaning; unsupplied MIME alternatives/attachments cannot be reconstructed. The example above is predicted legitimate. The SVM margin is not a probability.

## URL commands

```bash
python -m src.inference.url --url 'https://www.example.com' --json
python -m src.preprocessing.urls --verify
python -m src.features.url_pipeline --verify
python -m src.training.train_url --verify
python -m src.training.train_url --compare
python -m src.evaluation.evaluate_url --verify
```

URLs require HTTP(S), a valid dotted host/IP and port, no embedded whitespace, and at most 65,536 characters. Nothing is fetched or resolved. A Random Forest class estimate is not calibrated live risk. The model flags `https://example.com/` while accepting `https://www.example.com`, illustrating source-format sensitivity rather than destination safety.

## Interface and outputs

The terminal menu remains available:

```bash
python -m src.inference.predict --interactive
```

Choose `email`, `url` or `quit`. For email, finish multiline input with a single `.` line. The compatibility commands `python -m src.inference.predict email --body '...'` and `... url --url 'https://www.example.com'` still work. The local browser interface above uses the same saved models and preprocessing.

JSON includes `prediction`, `score`, `threshold`, `model`, `score_type`, `risk_level`, `score_note` and `risk_note`. Phishing means score >= the saved threshold: email −0.24541511286701478; URL 0.4754530853344011. Risk bands are provisional distances from the threshold, not calibrated confidence. CLI arguments may appear in shell history/process listings; interactive mode avoids body arguments.

## Reading the code

Each maintained Python file begins with a short explanation. A simple stage comment before every function describes its purpose. Task-specific logic stays in email or URL modules; common.py files contain shared utilities. URL-only external metrics are in `src/evaluation/evaluate_url.py`, and display-band widths are defined in each inference module. Training selection, preprocessing and saved thresholds are unchanged.

## Project files

```text
src/
  preprocessing/ emails.py, urls.py, common.py
  features/      email_features.py, email_pipeline.py
                 url_features.py, url_pipeline.py, common.py, pipeline.py
  training/      train_email.py, train_url.py, common.py, pipeline.py
  evaluation/    evaluate_email.py, evaluate_url.py, common.py, pipeline.py
  inference/     email.py, url.py, common.py, predict.py
  web/           app.py, index.html, __init__.py
tests/           test_email_*.py, test_url_*.py, shared contract helpers
models/          email/, url/, feature_engineering/
data/            raw_data/, processed/, splits/, features/ (local)
reports/         training/, evaluation/
docs/            three main guides; reference/email/ and reference/url/ (local)
```

| File or group | Responsibility |
|---|---|
| `src/preprocessing/emails.py` | MIME/CSV cleaning, duplicate/thread grouping, label checks and email splits |
| `src/preprocessing/urls.py` | URL syntax/canonical identity, label mapping, hostname splits and external exclusions |
| `src/preprocessing/common.py` | Shared content hash only |
| `src/features/email_features.py` | Frozen-compatible EmailFeatures class: TF-IDF/scaler; original serialized import path retained |
| `src/features/url_features.py` | Stateless 13-column lexical feature class; original definitions/order retained |
| `src/features/{email,url}_pipeline.py` | Independent feature export, alignment/schema validation and saved-run verification |
| `src/features/common.py` | Shared keyword counting, manifests and completion utilities |
| `src/training/train_email.py` | Email constructors, sparse loaders, preflight, verification and comparison |
| `src/training/train_url.py` | URL constructors, tabular loaders, preflight, verification and comparison |
| `src/training/common.py` | Generic fitting/scoring/selection utilities with task readers supplied explicitly |
| `src/evaluation/evaluate_email.py` | Email-only test loading, group overlap checks and selected-model evaluation |
| `src/evaluation/evaluate_url.py` | URL-only test/external loading, host/key overlap checks and evaluation |
| `src/evaluation/common.py` | Shared binary metrics, plots, error indices and evidence-preservation guards; URL-only external metrics live in evaluate_url.py |
| `src/inference/email.py`, `url.py` | Separate validation, retained-model loading, feature preparation and CLI |
| `src/web/app.py`, `index.html` | Local HTTP server, separate Email/URL forms and JSON endpoints using existing inference |
| `src/inference/common.py` | Shared artifact checks, scoring and formatting; task-specific display widths live in email.py and url.py |
| Remaining `pipeline.py`/`predict.py` facades | Backward-compatible dispatch/menu only, not duplicated task implementations |
| Package `__init__.py` files | Stable import structure |
| `tests/test_email_*.py`, `test_url_*.py` | Task-specific preprocessing, features, training, evaluation and inference tests |
| Shared test helpers/common tests | Task-neutral contracts, metric formulas, no-fit boundaries and bundle checks |
| `requirements.txt`, `requirements-dev.txt` | Pinned project dependencies; pytest separated for development |
| `environment.yml`, `pytest.ini`, `.gitignore` | Python environment, test discovery, local data/docs and cache exclusions |

Email artifacts include four original LR/SVM configurations, the selected relative alias, threshold and start marker. URL artifacts include LR/RF, selected alias, threshold and marker. `models/feature_engineering/` retains the email transformer and both schemas. Neither prediction requires the other task's model. Do not rename `src.features.email_features.EmailFeatures` or change feature order.

`data/processed/` contains cleaned Parquets, cleaning reports and the preserved email SQLite workspace; `data/splits/` contains frozen train/validation/test Parquets; `data/features/` contains email sparse NPZ/aligned row tables and URL lexical Parquets with manifests. Sources are Enron, two local email CSVs, PhiUSIIL and PhishTank. Original ZIP/checkpoint copies remain local and are not extra training data.

`reports/training/` preserves comparisons and completion evidence. `reports/evaluation/` preserves test/external metrics, confusion PNGs, error-row Parquets and run summaries. Shared global summaries are original run metadata, not mixed classifier implementations.

## Reproduce one task independently

Put that task's exact raw snapshots under a **fresh** root. Email expects `data/raw_data/emails/enron/maildir/` and `emails/nazario/{Nazario_5.csv,email_text.csv}`; URL expects `data/raw_data/urls/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv` and `urls/phishtank/online-valid.csv`. Current upstream feeds may differ from the original run, and exact email CSV lineage is unresolved.

```bash
REPRO_ROOT=/absolute/path/to/fresh-email-root
python -m src.preprocessing.emails --root "$REPRO_ROOT" --seed 42 --workers 4
python -m src.features.email_pipeline --root "$REPRO_ROOT" --seed 42
python -m src.training.train_email --root "$REPRO_ROOT" --seed 42
python -m src.evaluation.evaluate_email --root "$REPRO_ROOT"
```

```bash
REPRO_ROOT=/absolute/path/to/fresh-url-root
python -m src.preprocessing.urls --root "$REPRO_ROOT" --seed 42
python -m src.features.url_pipeline --root "$REPRO_ROOT" --seed 42
python -m src.training.train_url --root "$REPRO_ROOT" --seed 42
python -m src.evaluation.evaluate_url --root "$REPRO_ROOT"
```

Email preprocessing rebuilds outputs: do not run it on preserved evidence. Training fits only training; candidate selection and threshold tuning use validation; final evaluation uses the saved selected model/threshold on held-out data. Completed later stages verify/skip, and orphaned artifacts refuse automatic overwriting. `--compare --output /unused/file.json` creates a fresh validation comparison including accuracy/confusion/timing; it never fits or tunes. Original fit times are explicitly attributed, scoring time is measured anew.

## Preserved results and tests

| Task | Test rows | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Email | 44,947 | 0.996618 | 0.984090 | 0.983673 | 0.983881 |
| URL | 35,202 | 0.995995 | 0.999058 | 0.991519 | 0.995274 |

Email FP/FN: 75/77; URL FP/FN: 14/127. External PhishTank recall: 24,440/24,771 = 0.986638, with 331 misses. Positive-only data cannot establish false alerts. Saved original models and results are unchanged.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
```

Tests use temporary synthetic fits and frozen artifact checks; they do not retrain preserved models. Keep local datasets/candidate artifacts for full saved-run verification. A code-only clone can use selected pretrained artifacts and the maintained tests, but does not include all experimental inputs.

## Local documentation

`docs/` remains ignored by Git. Main guides are `ACADEMIC_REPORT.md`, `EXECUTION_GUIDE.md` and `DEFENSE_MASTER_GUIDE.md`. Task-specific pipelines, datasets, algorithms, evaluation and defense notes are separated under `docs/reference/email/` and `docs/reference/url/`. Generic metric/architecture/codebase references are shared. The current verification record is local under `docs/reference/audit/`.
