# Email preprocessing

Run from the repository root:

```bash
conda run -n phishing-ml python -m pip install -r requirements.txt
conda run --no-capture-output -n phishing-ml python -m src.preprocessing.emails --seed 42
conda run -n phishing-ml python -m pytest -q
```

`--root /absolute/repository/path` overrides automatic repository discovery. Source discovery scans only `data/raw_data/emails`, excluding notebook checkpoints. Missing sources are logged; absent sources are never fabricated. CSVs require `label` plus `text` or `body`. Only literal binary numeric labels are accepted. Files are read in sorted order for deterministic ingestion. CSV row numbers include the header.

Outputs are under `data/processed/emails` and `data/splits/emails`. The SQLite working store keeps content off the Python heap; it contains sensitive information and is ignored by Git along with the entire data directory. Repeated runs rebuild it and replace exports. Do not run two pipelines against the same root concurrently. A full run parses over half a million files and can take tens of minutes. Split ratios are approximate because groups are indivisible.

No attachments are executed, no remote resources are loaded, and HTML parsing is offline. Text attachments are excluded. Available headers are retained as JSON; repeated header names are collapsed to their last occurrence. Plain MIME alternatives are preferred for body text, and URLs from HTML alternatives are also retained. Explicit HTTP(S), FTP and www links are extracted with occurrence counts; bare domains and obfuscated links remain in text. Unicode NFC, HTML entity decoding and whitespace normalization preserve punctuation and case in model text. Grouping separately uses word normalization.

Exact duplicates use normalized body hashes, deliberately ignoring subject changes. Near-duplicate groups connect matching canonical bodies, substantial paragraphs (at least 80 words), quoted/forwarded blocks, and Message-ID/reference chains. This scalable heuristic can overgroup boilerplate and miss edited or partial quotations. It does not guarantee exhaustive semantic deduplication. All detected groups stay in one split. Entire conflicting-label groups are excluded; exact copies retain the first representative. Other near duplicates are retained in their shared split.

Provenance: the repository README cites https://www.kaggle.com/datasets/rohansood98/phishing-email-dataset-nazario-5-and-trec07. The local `Nazario_5.csv` has 3,065 structured records; `email_text.csv` has 53,668 pre-normalized text records. The original derivation of the latter and whether its positive class means spam, phishing, or both could not be established from available documentation. Numeric labels are preserved with `provided_binary_semantics_unverified`; this is not validated phishing ground truth. Enron records receive `presumed_legitimate`, never verified benign status. Retain this limitation in later evaluations or exclude unverified sources pending documentation review.

Use only text/URL content as candidate model features; source, paths, headers, duplicate IDs, original labels and label confidence can leak origin/target information. Compare source-specific metrics and consider an independent, verified contemporary holdout. Later training can use class weights and precision/recall or PR-AUC; never fit learned transformations or resampling on validation/test data. No model training is performed here.
