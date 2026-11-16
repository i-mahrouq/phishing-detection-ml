# Offline CLI inference

From the repository root:

```bash
conda run -n phishing-ml python -m src.inference.predict email --subject "Account verification" --body "Please verify your account at https://example.test/login" --json
conda run -n phishing-ml python -m src.inference.predict url --url "https://example.test/login?account=1" --json
conda run --no-capture-output -n phishing-ml python -m src.inference.predict --interactive
```

Global `--root` overrides artifact discovery. Omit `--json` for readable output.
Interactive email bodies accept multiple lines, terminated by a single `.` line.
Empty/invalid inputs produce errors; non-interactive errors exit with status 2.
Predictions use the full saved threshold precision and `score >= threshold`.

Email preprocessing reuses the exact existing cleaning and URL extraction
functions for the supplied subject/body, then calls the frozen TF-IDF/numerical
transformer. HTML links are counted before cleaning. Subject+body text and
numerical column order match ingestion/training. This interface takes a body,
not a raw MIME message; pass decoded plain or HTML content. MIME alternatives
and attachments are outside this interface.

URLs go through the existing stateless lexical transformer using the original
string and saved feature order. Nothing is fetched or resolved over the network.
Models load lazily and remain cached during interactive use. No fitting, dataset
processing, threshold updates, or persistent prediction logging occurs.

Provisional risk rules (heuristic, not calibrated or validated risk estimates):

| Model | LOW | MEDIUM | HIGH |
|---|---|---|---|
| Email SVM | score < threshold - 0.5 | between the two bounds | score >= threshold + 0.5 |
| URL RF | score < threshold - 0.10 | between the two bounds | score >= threshold + 0.10 |

Risk bands do not change the classification threshold. A prediction near the
threshold can be MEDIUM on either side. SVM margins are never probabilities or
percentages. Random Forest scores are model-estimated probabilities, not
guaranteed real-world risk. Existing data-source and label-confidence limitations
still apply. Load only trusted local joblib artifacts.
