# Feature engineering

Run from the repository root in `phishing-ml`:

```bash
python -m src.features.pipeline --seed 42
python -m src.features.pipeline --verify
```

No preprocessing or classifier training is invoked. Existing completed feature outputs are verified and skipped. Errors stop execution; there is no automatic retry. Preserve preprocessing label-confidence limitations when using these features.

Email outputs: `data/features/emails/{train,validation,test}.npz` (float32 CSR, TF-IDF followed by scaled numerical columns) and matching `{split}_rows.parquet` (`record_id`, `label`, original row order). Fit is performed once on training only. Word unigrams/bigrams use a maximum 50,000 terms, minimum document frequency 2, and sublinear term frequencies. Character n-grams are omitted to bound memory and runtime. Numbers use train-fitted MaxAbsScaler; inference values may exceed one.

URL outputs: `data/features/urls/{train,validation,test,external}.parquet`. Only the columns listed in the saved schema are predictive. `url_key` and `label` are alignment/target metadata and must be excluded from model inputs. Original URL strings provide lexical evidence; no supplied webpage features are used. PhishTank external data is transformed without fitting. Subdomain counts use a documented label-count heuristic, not an online public-suffix lookup.

Transformers/schema: `models/feature_engineering/email_transformer.joblib`, `emails_schema.json`, `urls_schema.json`. Each dataset has a `feature_report.json` success marker recording input file metadata and validation results. Verification checks unchanged input splits, exact row/label alignment, consistent columns, finite values, and persisted email transformer schema. Generated data and fitted vocabularies are ignored by Git.

Inference (use only trusted local joblib files):

```python
import joblib
import pandas as pd
from src.features.url_features import URLFeatures

email = joblib.load('models/feature_engineering/email_transformer.joblib')
X = email.transform(pd.DataFrame([{
    'text': 'Urgent account verification https://example.test/login',
    'subject': 'Urgent account verification', 'url_count': 1,
}]))
U = URLFeatures().transform(['https://example.test/login?account=1'])
```

Email `text` must use the cleaned subject-plus-body representation from preprocessing. Optional subject and URL count default to an empty subject and explicit links found in text. Supply MIME-derived counts to include HTML-alternative links, just as the existing email splits do. Labels and source identifiers never enter the transformer. Keep the same available metadata policy at inference.
