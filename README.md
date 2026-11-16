# Phishing Detection — Machine Learning

A reproducible academic cybersecurity project with two pretrained, offline classifiers: **Email V1** detects suspicious message content; **URL V1** classifies lexical URL structure. A local **Streamlit** interface and a CLI reuse the same frozen models, feature transformations and decision thresholds.

**Phishing** impersonates trusted organizations or people to steal credentials, money or other sensitive information. Machine learning can help prioritize suspicious inputs, but these models are research demonstrations. High internal scores do not establish real-world safety. In particular, the URL dataset contains strong **source-format bias**. Never use a prediction as permission to open a link.

## System architecture

```text
Raw datasets → Preprocessing → Feature engineering → Training → Evaluation
                                                          ↓
                                               Saved models + thresholds
                                                          ↓
                                            CLI / Streamlit prediction
```

| Responsibility | Implementation | Purpose |
|---|---|---|
| Email ingestion | `src/preprocessing/emails.py` | Parse MIME/CSV, clean text, deduplicate and group-split |
| URL ingestion | `src/preprocessing/urls.py` | Validate URLs offline, map labels, canonicalize duplicate keys and hostname-split |
| Feature generation | `src/features/pipeline.py` | Fit only training transforms; preserve split/label alignment |
| Email features | `src/features/email_features.py` | Sparse TF-IDF and scaled numerical metadata |
| URL features | `src/features/url_features.py` | Thirteen ordered lexical measurements |
| Shared helpers | `src/features/common.py` | Suspicious-term matching and metadata utilities |
| Training/selection | `src/training/pipeline.py` | Compare configurations and select validation thresholds |
| Final evaluation | `src/evaluation/pipeline.py` | Frozen-threshold internal test and positive-only external metrics |
| Inference | `src/inference/predict.py` | Load trusted artifacts lazily and predict without datasets |
| Interface | `app.py` | Streamlit forms calling the existing Predictor |

Training and evaluation are separate from prediction. Starting the application never downloads data, trains a classifier or fits TF-IDF.

## Repository structure

```text
phishing-detection-ml/
├── app.py
├── .streamlit/config.toml
├── src/
│   ├── preprocessing/
│   ├── features/                 # preserved module path for serialized transformer
│   ├── training/
│   ├── evaluation/
│   └── inference/
├── scripts/verify_pretrained.py
├── models/
│   ├── email/                    # selected SVM, reference symlink, threshold
│   ├── url/                      # selected forest, reference symlink, threshold
│   └── feature_engineering/      # fitted email transformer and both schemas
├── data/
│   ├── raw_data/README.md        # local data excluded from Git
│   ├── processed/README.md
│   ├── splits/README.md
│   └── features/README.md
├── reports/{training,evaluation}/ # small V1 aggregate results
├── tests/
├── docs/                         # datasets, cleanup, research and verification
├── requirements.txt
├── requirements-dev.txt
└── environment.yml
```

Local `archive/research/`, experimental models/reports and notebooks are preserved but excluded from the active repository. No local dataset was removed. V2/V3 were previous experimental research; see [research history](docs/research_history.md) and [cleanup plan](docs/cleanup_plan.md).

## Datasets

The application uses **0 = legitimate, 1 = phishing**. Email V1 combines historical Enron messages (assumed legitimate) with supplied numeric labels from `Nazario_5.csv` and `email_text.csv`. Their exact CSV lineage and phishing-versus-spam label semantics remain unverified. URL V1 uses PhiUSIIL, whose original labels are reversed: original 1 becomes legitimate 0, original 0 becomes phishing 1. PhishTank supplies a phishing-only external check, not benign examples.

| Dataset after cleaning | Legitimate | Phishing | Total |
|---|---:|---:|---:|
| Email V1 | 268,209 | 31,443 | 299,652 |
| URL V1 (PhiUSIIL) | 134,849 | 99,838 | 234,687 |
| External PhishTank after overlap exclusions | 0 | 24,771 | 24,771 |

See [dataset documentation](docs/datasets.md) for official upstream links, unresolved provenance, exact paths, schemas, class mappings, real redacted examples and acquisition instructions. Directory guides: [raw](data/raw_data/README.md), [processed](data/processed/README.md), [splits](data/splits/README.md), [features](data/features/README.md). Dataset licenses and access conditions remain those of their original providers; this repository does not redistribute the corpora.

## Data preprocessing and leakage prevention

Email preprocessing decodes MIME, extracts links before stripping HTML, removes script/style content, normalizes Unicode/whitespace and discards unusable bodies. It keeps case and punctuation. Exact body duplicates are removed; shared long paragraphs/quoted threads form conservative duplicate groups. Conflicting-label groups are excluded. A seeded group-aware split targets 70/15/15; it cannot guarantee semantic independence. Implementation: `src/preprocessing/emails.py`.

URL preprocessing accepts HTTP(S) URLs with valid host/port syntax and preserves the original URL for lexical features. A separate normalized value/canonical key supports deduplication; source and label are never predictive columns. Missing/invalid records, canonical duplicates and conflicting labels are excluded. Twenty stratified **hostname-group** folds are allocated 14/3/3 with seed 42. Related registered domains can still cross splits. Implementation: `src/preprocessing/urls.py`.

**Data leakage** means training or selection receives information from evaluation data. Learned vocabulary, IDF and scaling are fitted on training only; validation selects models/thresholds; the test split evaluates the frozen result. Existing splits remain unchanged. Duplicate grouping and hostname splitting reduce specific leakage routes, but do not prove independence from source bias or related domains.

| Split | Email rows (legitimate / phishing) | URL rows (legitimate / phishing) |
|---|---:|---:|
| Train | 209,758 (187,747 / 22,011) | 164,283 (94,396 / 69,887) |
| Validation | 44,947 (40,231 / 4,716) | 35,202 (20,226 / 14,976) |
| Test | 44,947 (40,231 / 4,716) | 35,202 (20,227 / 14,975) |

## Feature engineering

### Email: word TF-IDF and six numerical features

**TF-IDF** weights terms by within-message frequency and inverse corpus frequency, so corpus-common terms carry less information. **Unigrams** are individual words; **bigrams** are adjacent word pairs that capture phrases. `EmailFeatures` in `src/features/email_features.py` uses word unigrams/bigrams, a fitted vocabulary of 50,000 terms, `min_df=2`, sublinear term frequency, and L2 text normalization.

Six numeric columns follow the vocabulary in this exact order: `text_length`, `word_count`, `url_count`, `suspicious_keyword_count`, `subject_length`, `exclamation_count`. They represent message size, link density, suspicious terms and punctuation/subject structure. Suspicious terms are heuristic signals, not proof of malicious intent.

**Feature scaling** gives these numeric columns comparable ranges: a training-fitted **MaxAbsScaler** divides each by its training maximum absolute value. New inputs can exceed one. **Sparse CSR matrices** store nonzero entries without allocating every vocabulary position: the selected SVM expects **50,006 float32 features**. Text features remain sparse throughout. `src/features/pipeline.py` saves fitted transforms, schemas, matrices and row identifiers; inference calls only `transform`.

### URL: thirteen lexical features

**Lexical features** measure the URL string itself without DNS or HTTP requests. `src/features/url_features.py` produces the following ordered float32 columns, without feature scaling for the selected forest:

| Feature | Meaning |
|---|---|
| `url_length` | Characters in the original URL |
| `hostname_length` | Characters in parsed hostname |
| `dot_count` | Dots anywhere in the URL |
| `subdomain_count` | max(hostname label count − 2, 0), zero for IPs; heuristic, not public-suffix aware |
| `digit_count` | Unicode digits in the original string |
| `hyphen_count` | Hyphens in the original string |
| `special_character_count` | Non-alphanumeric characters including scheme punctuation |
| `is_https` | Whether parsed scheme is HTTPS |
| `hostname_is_ip` | Whether hostname parses as an IP address |
| `url_entropy` | Shannon entropy of original characters |
| `suspicious_keyword_count` | Suspicious substring occurrences |
| `path_length` | Characters in parsed path |
| `query_parameter_count` | Nonempty ampersand-separated query components |

Long, irregular or keyword-rich URLs can expose obfuscation. Legitimate services can exhibit the same properties. HTTPS means encrypted transport, **not trustworthiness**. Formatting measurements can become shortcuts for dataset origin; feature definitions/order are preserved to keep pretrained behavior unchanged. Real stored feature values appear in [datasets.md](docs/datasets.md#feature-datasets-and-real-values).

## Model training and selection

**Linear SVM** learns a separating hyperplane in the high-dimensional sparse email space. Its signed decision margin is not a calibrated probability. **Random Forest** aggregates decision trees and captures nonlinear lexical interactions; its class estimates are not guaranteed real-world probabilities.

Email candidates were logistic regression (`liblinear`, `max_iter=1000`) and LinearSVC (`dual='auto'`, `max_iter=10000`), each with no class weighting and balanced weighting. URL candidates were logistic regression with a training-fitted StandardScaler and a Random Forest (`n_estimators=100`, `min_samples_leaf=3`, `n_jobs=2`). Random seeds are 42. **Class imbalance** makes accuracy alone misleading, especially with predominantly legitimate email; balanced weights were evaluated rather than assumed optimal.

Configurations were ranked by validation **average precision**, then phishing F1, then recall at the default threshold. For the selected configuration, the validation threshold maximized phishing F1, with recall breaking ties. No test/external data selected the threshold.

| Detector | Selected configuration | Frozen threshold | Score |
|---|---|---:|---|
| Email V1 | `linear_svm_none` (class_weight=None) | -0.24541511286701478 | SVM decision margin |
| URL V1 | `random_forest` | 0.4754530853344011 | Class-1 forest estimate |

Both apply exactly **phishing if score >= threshold**. A **decision threshold** trades false alarms against missed phishing; the included thresholds must not be interpreted as calibrated safety cutoffs.

Included artifacts:

- `models/email/linear_svm_none.joblib`, `selected_model.joblib` (relative symlink), `selected_threshold.json`.
- `models/url/random_forest.joblib`, `selected_model.joblib` (relative symlink), `selected_threshold.json`.
- `models/feature_engineering/email_transformer.joblib`, `emails_schema.json`, `urls_schema.json`.

All individual artifacts are under GitHub's 100 MiB per-file limit. Preserve symlinks when copying manually; normal Linux Git clones retain them. Only load trusted joblib files: Python serialization can execute code. See `reports/training/*_model_comparison.json` for original configuration results, timings and selection metadata.

## Model evaluation

These are **existing frozen V1 test results**, recovered from saved reports; cleanup did not rerun large evaluations.

| Metric | Email V1 (44,947) | URL V1 (35,202) |
|---|---:|---:|
| Accuracy | 0.996618 | 0.995995 |
| Phishing precision | 0.984090 | 0.999058 |
| Phishing recall | 0.983673 | 0.991519 |
| Phishing F1 | 0.983881 | 0.995274 |
| Average precision | 0.998388 | 0.998074 |
| ROC-AUC | 0.999809 | 0.997536 |
| False-positive rate | 0.001864 (0.1864%) | 0.000692 (0.0692%) |
| TN / FP / FN / TP | 40,156 / 75 / 77 / 4,639 | 20,213 / 14 / 127 / 14,848 |

**Accuracy** is the fraction correct. **Precision** is TP/(TP+FP): how many alerts are phishing. **Recall** is TP/(TP+FN): how much phishing is detected. **F1-score** is their harmonic mean. A **confusion matrix** shows legitimate correctly accepted (TN), legitimate flagged (FP), phishing missed (FN), and phishing detected (TP). **Average precision** summarizes precision-recall ranking across thresholds and is useful for imbalanced classes; it is not a single-threshold accuracy. **False-positive rate** is FP/(TN+FP), the share of legitimate inputs flagged. **ROC-AUC** summarizes true-positive versus false-positive ranking across thresholds.

On the previously inspected, positive-only PhishTank snapshot: **24,440/24,771 detected**, **331 missed**, recall **0.986638**. Without legitimate examples this benchmark cannot estimate precision or FPR. Internal metrics do not establish generalization. See [email metrics](reports/evaluation/email_test_metrics.json), [URL metrics](reports/evaluation/url_test_metrics.json), [PhishTank metrics](reports/evaluation/phishtank_external_metrics.json), and the saved confusion plots in `reports/evaluation/`.

## Installation after cloning (Ubuntu/Linux)

Use **Python 3.11**. Dependency versions match the environment that loads the saved artifacts; scikit-learn pickle compatibility is version-sensitive.

```bash
git clone https://github.com/i-mahrouq/phishing-detection-ml.git
cd phishing-detection-ml
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m scripts.verify_pretrained
```

If your Ubuntu installation lacks Python 3.11/venv support, install it through your supported Python distribution first. `environment.yml` is an optional Conda equivalent (`conda env create -f environment.yml`, then `conda activate phishing-ml`). Selected models and fitted transforms are included: **datasets and retraining are not required for prediction**. `verify_pretrained` prints two example results while validating model classes, schemas, feature dimensions, references, labels and thresholds; a suspicious example.com verdict is possible and illustrates source bias.

### CLI predictions

```bash
python -m src.inference.predict email --subject "Meeting" --body "Project meeting tomorrow at ten." --json
python -m src.inference.predict url --url "https://www.example.com" --json
python -m src.inference.predict --interactive
```

The CLI accepts a subject and body, not a raw .eml filename. Email processing counts links before cleaning subject+body. CLI risk bands are provisional native-score heuristics, not calibrated confidence. Use the saved classification and examine its limitations.

### Web Interface — How to Use

```bash
streamlit run app.py
```

1. Open **http://localhost:8501** in a browser (the server binds locally).
2. Select **Email detection** or **URL detection**.
3. Paste an email body (optional subject), or an HTTP(S) URL.
4. Click **Analyze Email** or **Analyze URL**.
5. Read **Phishing** / **Legitimate** and the visible limitation notice. Expand model details for raw score/threshold; URL results also offer the thirteen lexical values.
6. Press **Ctrl+C** in the launching terminal to stop.

Safe inputs include subject `Meeting` with body `Project meeting tomorrow at ten.`, and `https://www.example.com`. For a suspicious illustrative message, use `Urgent: verify your account at https://example.com/login`. These examples are synthetic inputs, **not measured accuracy or guaranteed verdicts**; example.com is a reserved documentation domain. Submitted HTML is never rendered or executed. Links are never opened. Inputs remain in the local Streamlit session, are not sent to external APIs and are not persistently logged by application code. Streamlit usage telemetry is disabled in `.streamlit/config.toml`. Keep this local development interface off public networks.

## Prediction examples

Observed frozen-model outputs for safe illustrative inputs (not benchmark results):

| Scenario | Input | Observed output |
|---|---|---|
| Legitimate-style email | Subject: Meeting; body: Project meeting tomorrow at ten. | Legitimate |
| Phishing-style email | Subject: Urgent account verification; body: Your account will be suspended. Verify your password immediately at https://example.com/login to restore access. | Phishing |
| Legitimate-style URL | `https://www.example.com` | Legitimate |
| Phishing-style URL | `http://192.0.2.1/login?verify=account` | Phishing |

The IP address is reserved for documentation; no URL was visited. Changing the benign example to `https://example.com/` produced **Phishing**, a concrete false-positive illustration of formatting bias, not evidence that example.com is malicious.

## Reproduce the complete pipeline

Reproduction is separate from pretrained use. Follow [dataset download/placement instructions](docs/datasets.md#raw-datasets-and-placement) and resolve the two CSV provenance/label uncertainties before claiming an independently reproducible corpus. Preserve historical originals; fresh snapshots can differ. In a **separate working copy** with raw inputs and no completed generated stage outputs:

```bash
# Run from this repository; place original datasets below this separate root first.
REPRO_ROOT="$HOME/phishing-ml-reproduction"
python -m src.preprocessing.emails --root "$REPRO_ROOT" --seed 42 --workers 4
python -m src.preprocessing.urls --root "$REPRO_ROOT" --seed 42
python -m src.features.pipeline --root "$REPRO_ROOT" --seed 42
python -m src.training.pipeline --root "$REPRO_ROOT" --seed 42
python -m src.evaluation.pipeline --root "$REPRO_ROOT"
python -m src.inference.predict --root "$REPRO_ROOT" url --url "https://www.example.com"
```

The separate root must have `data/raw_data/` laid out as documented, and no generated outputs. The shipped selected models are an inference bundle, not a complete training checkpoint. These commands create cleaned Parquets/reports, fixed splits, sparse/dense lexical feature artifacts, models/validation reports, and final evaluation reports respectively. Existing completed stages verify/skip; ambiguous partial training stops. Do not delete local datasets or shipped artifacts to force a rerun. Training can require substantial RAM/time; it is never part of application startup. See the per-module READMEs for details. Training/test/external boundaries must remain intact.

## Tests and verification

```bash
pip install -r requirements-dev.txt
python -m pytest tests/test_inference.py tests/test_evaluation.py tests/test_app.py -q
```

The cleanup verification passed **14 targeted tests** (8 inference/evaluation/UI/packaging tests and 6 preprocessing/URL-feature checks); see [verification and artifact checksums](docs/verification.md). These lightweight checks load frozen artifacts, verify preprocessing/score parity, input validation, threshold semantics, UI/CLI consistency, blocked network access and a Git-index copy containing no dataset files. The packaging test requires the active files to be staged/tracked (normal clones already satisfy this). Additional preprocessing/features/training tests use tiny synthetic fixtures, not full corpora; training tests are intentionally excluded from the smoke command. `.gitignore` excludes all raw, processed, split and feature data while keeping dataset README files, selected model binaries, schemas and small final reports.

## Limitations

- **Dataset/source bias:** historical corporate email and uncertain mixed CSV labels can overstate phishing-specific performance.
- **URL source-format bias:** every raw PhiUSIIL legitimate URL uses HTTPS, www and no trailing slash/path. Ordinary benign formatting changes can trigger false positives. URL V1 is **not production-ready**.
- **Leakage risk:** group/hostname splitting cannot guarantee registered-domain, temporal or semantic independence; source patterns can leak label information without an implementation bug.
- Internal test accuracy and a phishing-only external recall do not establish real-world reliability or benign external FPR.
- SVM margins and forest estimates are uncalibrated. Models can miss new attacks and flag legitimate messages/URLs. A legitimate prediction is not proof of safety.
- No webpage fetching, attachment inspection, reputation service, live threat verification or multilingual coverage guarantee.
- Exact historical corpus regeneration is limited by upstream snapshot availability and unresolved CSV lineage. Data/provider licenses do not imply an unrestricted right to redistribute pretrained artifacts; review applicable terms before wider distribution.

## Future improvements

Acquire representative, reliably labeled contemporary full URLs and emails; evaluate source-aware and registered-domain-disjoint splits; preserve genuinely unseen external and temporal benchmarks; measure robustness to ordinary formatting changes. Consider calibration and a reviewed API/web deployment only after independent validation. V2/V3 experiments informed these priorities and remain local historical research; they are outside the active V1 pipeline.
