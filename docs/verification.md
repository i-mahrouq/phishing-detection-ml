# Verification and packaging

Python 3.11 environment: 8 inference/evaluation/UI/packaging tests and 6 targeted preprocessing/URL-feature tests passed; 14 total. No training or feature generation was run. pip check: no broken requirements.

All 59 baseline paths retained original size/mtime; 15 V1 artifact hashes matched. This is a targeted preservation check, not a new full-corpus hash scan. All dataset changes are README additions. Experimental files were moved, not deleted.

The dataset-free simulation copies only Git-index files into a temporary directory, preserves relative model symlinks and runs both CLI and Streamlit predictions. It uses the installed, pinned environment; it is not a new OS or clean-environment installation.

## Included frozen artifact SHA-256 values

| Artifact | SHA-256 |
| --- | --- |
| `models/email/linear_svm_none.joblib` | `97a63ae7a8261dc978058453dea892ee35ff3b680652ad3f4934cda872b68076` |
| `models/email/selected_threshold.json` | `eb409230c6a1482dbd11b0e56df549b83880240a4e54ad376af7d04c9bd28bf7` |
| `models/feature_engineering/email_transformer.joblib` | `5fc20fba6bb8de1245bc316d56bcf7ecc1110fb49d76b2f4404f7d5c97f4e1da` |
| `models/feature_engineering/emails_schema.json` | `0532e1b2326703d782141887d4f4c04e40321c4b3f3972b026896b9ed1801720` |
| `models/feature_engineering/urls_schema.json` | `1b04df11180c23ac600d7c80984304ab5fec43041f3ebb49cb8a85e95458362c` |
| `models/url/random_forest.joblib` | `c65f2d926e8bd7579fc9cd1d93d26022268d699e031b7b04f11d88eb9c1eb6b6` |
| `models/url/selected_threshold.json` | `8c22794d3f7616a3af95f4c3ade7198448326c10d3abcb148fc47f04592066a3` |

No dataset file is staged; four dataset README files are included. Selected binaries are below GitHub's 100 MiB limit. No commit/push/history rewrite was performed.

## Remaining limitations

Two local email CSVs have unresolved original provenance/usage terms and label semantics. Research metrics and trained artifacts cannot establish live safety. No new project license is asserted; owner/provider licensing review is needed before granting redistribution rights. Historical reproduction needs the exact local snapshots; currently available upstream versions can differ.
