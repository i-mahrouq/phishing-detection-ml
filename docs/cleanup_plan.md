# Cleanup plan and scope

The active application is Email V1 (Linear SVM) and URL V1 (Random Forest). No model is retrained, no transformer is fitted, and no label, feature, threshold or split semantics are changed.

| Classification | Files | Action |
|---|---|---|
| Keep | V1 modules under src/preprocessing, src/features, src/training, src/evaluation and src/inference; common/__init__ modules | Retain original module names for pickle/import compatibility; add focused documentation/type hints only |
| Keep | models/email/linear_svm_none.joblib; models/url/random_forest.joblib; selected_model.joblib symlinks; both selected_threshold.json files; email_transformer.joblib and emails_schema.json/urls_schema.json | Include explicitly in Git, preserve bytes and symlink targets |
| Keep | V1 test metrics/confusion plots and training comparisons | Include small aggregate reports; leave error-row Parquets local/ignored |
| Keep | Every raw, processed, split and feature dataset, including experimental data and archives | Keep at the original local path; exclude data files from Git, never delete or rewrite |
| Reorganize | V2/V3-specific Python modules, scripts, tests and their documentation | Move to ignored archive/research with original relative layout; keep as historical research, outside active test discovery |
| Reorganize | Historical dependency lock/environment snapshots | Preserve original copies in archive/research; active requirements contain V1/UI dependencies only |
| Remove from active scope | Experimental entry points, candidates, large intermediate reports, logs/caches | Ignore locally; do not delete research artifacts or datasets |
| Uncertain—preserve | User-authored notebooks, old archives, duplicate source copies, experimental research | Preserve all locally; ignore notebooks/archives pending privacy/relevance review. No deletion authorized or performed |

## Structure and safety

Keep src/features (no forced rename): the serialized email transformer imports src.features.email_features.EmailFeatures. Existing V1 commands and artifact paths remain stable. Add app.py as a local Streamlit client of Predictor, not a separate classifier. User inputs are neither fetched nor logged.

Git initially tracked only README.md; no dataset untracking or history rewrite is required. Stage active code/tests/docs, small aggregate reports and an explicit model allowlist. Do not commit, push, rewrite history or delete local files. Archive remains local and ignored; historical imports are not maintained as active runnable pipelines.

Verification: selected artifact SHA-256/threshold equality; metadata checks on known local datasets; targeted V1/UI tests; fresh-clone simulation containing only Git-selected files and no datasets; Git ignore/size checks. No full training, feature generation or large evaluations.

## Executed archival moves

- `src/preprocessing/urls_v2.py` → `archive/research/src/preprocessing/urls_v2.py`
- `src/features/url_features_v2.py` → `archive/research/src/features/url_features_v2.py`
- `src/features/urls_v2_pipeline.py` → `archive/research/src/features/urls_v2_pipeline.py`
- `src/training/urls_v2.py` → `archive/research/src/training/urls_v2.py`
- `config/url_features_v2.v1.json` → `archive/research/config/url_features_v2.v1.json`
- `scripts/evaluate_url_v2.py` → `archive/research/scripts/evaluate_url_v2.py`
- `scripts/qualify_url_v3.py` → `archive/research/scripts/qualify_url_v3.py`
- `scripts/report_url_v3_audit.py` → `archive/research/scripts/report_url_v3_audit.py`
- `scripts/acquire_url_v3.py` → `archive/research/scripts/acquire_url_v3.py`
- `scripts/report_url_v2_diagnostics.py` → `archive/research/scripts/report_url_v2_diagnostics.py`
- `scripts/predict_url_v2.py` → `archive/research/scripts/predict_url_v2.py`
- `scripts/audit_url_v3.py` → `archive/research/scripts/audit_url_v3.py`
- `scripts/select_url_v2.py` → `archive/research/scripts/select_url_v2.py`
- `scripts/diagnose_url_v2.py` → `archive/research/scripts/diagnose_url_v2.py`
- `scripts/report_url_v3_qualification.py` → `archive/research/scripts/report_url_v3_qualification.py`
- `tests/test_url_evaluation_v2.py` → `archive/research/tests/test_url_evaluation_v2.py`
- `tests/test_url_training_v2.py` → `archive/research/tests/test_url_training_v2.py`
- `tests/test_url_diagnostics_v2.py` → `archive/research/tests/test_url_diagnostics_v2.py`
- `tests/test_urls_v2.py` → `archive/research/tests/test_urls_v2.py`
- `tests/test_url_cli_v2.py` → `archive/research/tests/test_url_cli_v2.py`
- `tests/test_url_selection_v2.py` → `archive/research/tests/test_url_selection_v2.py`
- `tests/test_url_features_v2.py` → `archive/research/tests/test_url_features_v2.py`
- `tests/test_url_v3_dataset_audit.py` → `archive/research/tests/test_url_v3_dataset_audit.py`
- `tests/test_url_v3_qualification.py` → `archive/research/tests/test_url_v3_qualification.py`
- `docs/url_v2_cli.md` → `archive/research/docs/url_v2_cli.md`
- `docs/url_v2_selection.md` → `archive/research/docs/url_v2_selection.md`
- `docs/url_v2_preprocessing.md` → `archive/research/docs/url_v2_preprocessing.md`
- `docs/url_v2_evaluation.md` → `archive/research/docs/url_v2_evaluation.md`
- `docs/url_v2_training.md` → `archive/research/docs/url_v2_training.md`
- `docs/url_v2_feature_engineering.md` → `archive/research/docs/url_v2_feature_engineering.md`

## Completed checks

See [verification](verification.md): 14 targeted tests passed, including frozen inference, UI/CLI parity, invalid inputs, offline prediction and a Git-index copy with no datasets. Fifteen V1 artifact hashes matched the baseline; 59 original paths kept size/mtime. Only dataset README files are staged under data/. No commit or push was made. No source dataset, model, report or user notebook was deleted.
