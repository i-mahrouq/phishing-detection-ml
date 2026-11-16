# Datasets, provenance and reproducibility

All dataset files stay local and are excluded from Git. This document contains bounded real excerpts, not redistributed corpora. **0 = legitimate; 1 = phishing** is the application's convention; label reliability differs by source. Counts below come from saved cleaning/feature reports and Parquet metadata, not new full scans. Existing splits were preserved.

## Raw datasets and placement

| Dataset | Original provider / official source | Local path under data/raw_data/ | Format / raw records | Interpretation and limitations |
|---|---|---|---|---|
| Enron email | [CMU/CALO Enron collection](https://www.cs.cmu.edu/~enron/) | emails/enron/maildir/ | RFC822/MIME files; 517,401 discovered; 517,389 parsed | Assigned 0 from source membership; historical corporate mail is presumed legitimate, not independently verified benign. Privacy and era bias. The official corpus omits attachments. |
| Nazario_5.csv | Local repackaged CSV; original CSV publisher and exact lineage unverified. [José Nazario's original phishing corpus](https://monkey.org/~jose/phishing/) is historical context, **not verified provenance for this mixed-label CSV** | emails/nazario/Nazario_5.csv | CSV; 3,065 | Existing numeric 0/1 retained. Some labels may represent spam rather than phishing. Filename alone establishes neither provenance nor semantics. |
| email_text.csv | Local repackaged CSV; original publisher / official matching download unverified | emails/nazario/email_text.csv | CSV; 53,668 | Existing numeric 0/1 retained with the same unresolved phishing-versus-spam semantics. Do not claim independently verified labels. |
| PhiUSIIL | [UCI official dataset](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset) | urls/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv | CSV; 235,795; original 134,850 legitimate / 100,945 phishing | Original label **1→legitimate 0; 0→phishing 1**. Strong source-format bias: every raw legitimate URL uses HTTPS + www and no path/trailing slash. CC BY 4.0; follow UCI attribution. |
| PhishTank | [Official PhishTank](https://phishtank.org/) / [developer information](https://phishtank.org/developer_info.php) | urls/phishtank/online-valid.csv | CSV; 45,341; phishing-only snapshot | Provider/community-verified online phishing at collection time →1. Historical URLs may expire; independent status is not rechecked. Positive-only external evaluation, never a benign source. Follow provider feed access/usage terms. |

Obtain the Enron archive through CMU and extract its maildir at the exact path above. Obtain PhiUSIIL from UCI, extract its CSV, retain its attribution. Obtain PhishTank's official verified-online CSV through its documented feed access process and place it as online-valid.csv. Never open listed phishing URLs. Sources change: a current download will not reproduce a historical snapshot's exact counts/checksums.

The two email CSVs were extracted from local emails/nazario/archive.zip (both CSV members, no additional examples). The archive is a duplicate container, not an extra dataset. Their exact matching original download and redistribution terms remain unresolved: preserve local originals and document verified provenance before sharing them. Do not replace them silently with a different Nazario MBOX or invert labels based on a guessed origin. No dataset license is granted by this repository. Enron may contain sensitive historical correspondence; examples below redact addresses and identity-bearing headers.

Other local datasets (OpenPhish, Tranco, urls_v3 crawl/collection samples) belong to experimental research, are preserved/ignored, and are **not V1 training inputs**. Popularity/crawl membership does not establish legitimacy. See [research history](research_history.md).

## Real raw examples and schemas

Examples retain source row identities. All long values are truncated (…); email addresses, secret-looking assignments and long identifiers are masked; sender/receiver headers are omitted. URL schemes/domains are defanged and query values suppressed. These are dataset labels, not a new safety assessment. No content was fetched.

### Enron

MIME headers are strings (Subject, From, To, Date, Message-ID, References); body is decoded text/plain or HTML. Attachments are not executed. File path identifies one message; the label below is added during ingestion.

| Original maildir path | Subject excerpt | Raw body excerpt | Ingestion label |
| --- | --- | --- | --- |
| allen-p/_sent_mail/1. |  | Here is our forecast ⏎  ⏎   | 0 (source-inferred) |
| allen-p/_sent_mail/100. | Re: test | test successful.  way to go!!! | 0 (source-inferred) |
| allen-p/_sent_mail/1000. |  | [name redacted], ⏎  ⏎  Can you send me a schedule of the salary and level of everyone in the  ⏎ scheduling group.  Plus your … | 0 (source-inferred) |

### Nazario_5.csv

Original columns: `sender`, `receiver`, `date`, `subject`, `body`, `label`, `urls`. CSV fields are read as text, except `label` parsed as integer 0/1. sender/receiver identify correspondents (omitted here); date is the supplied timestamp; subject/body contain content; urls is supplied link metadata; label is the supplied class. Row numbers below follow csv.DictReader record enumeration, starting at 2 after the header; embedded newlines do not define separate records.

| Source record | Original label | Subject excerpt | Original content excerpt |
| --- | --- | --- | --- |
| 5 | 0 | FW: ENA Upstream Company information | [name redacted], ⏎  ⏎ We are currently trading under GTC Spot contracts with El Paso for ENA Upstream.  Would you prefer to r… |
| 15 | 0 | FW: Mariner Management Meeting Minutes |  ⏎  ⏎  -----Original Message----- ⏎ From: 	"[name redacted]" <[email redacted]> ⏎ Sent:	Friday, June 29, 2001 9:27 AM ⏎ To:	[… |
| 1503 | 1 | Verify Your Account | Business with  								cPanel & WHM 								Dear client, Our Technical  								Services Department are carrying out a planne… |

### email_text.csv

Original columns: `label`, `text`. CSV fields are read as text, except `label` parsed as integer 0/1. text contains message content; label is the supplied class. Row numbers below follow csv.DictReader record enumeration, starting at 2 after the header; embedded newlines do not define separate records.

| Source record | Original label | Subject excerpt | Original content excerpt |
| --- | --- | --- | --- |
| 2 | 1 | (not supplied) | do you feel the pressure to perform and not rising to the occasion try v ia gr a your anxiety will be a thing of the past and… |
| 3 | 0 | (not supplied) | hi i've just updated from the gulus and i check on other mirrors it seems there is a little typo in debian readme file exampl… |
| 10 | 0 | (not supplied) | hi i have to use r to find out the escapenumber confidence interval for the sensitivity and specificity of the following diag… |

### PhiUSIIL

Original schema: `FILENAME`, `URL`, `URLLength`, `Domain`, `DomainLength`, `IsDomainIP`, `TLD`, `URLSimilarityIndex`, `CharContinuationRate`, `TLDLegitimateProb`, `URLCharProb`, `TLDLength`, `NoOfSubDomain`, `HasObfuscation`, `NoOfObfuscatedChar`, `ObfuscationRatio`, `NoOfLettersInURL`, `LetterRatioInURL`, `NoOfDegitsInURL`, `DegitRatioInURL`, `NoOfEqualsInURL`, `NoOfQMarkInURL`, `NoOfAmpersandInURL`, `NoOfOtherSpecialCharsInURL`, `SpacialCharRatioInURL`, `IsHTTPS`, `LineOfCode`, `LargestLineLength`, `HasTitle`, `Title`, `DomainTitleMatchScore`, `URLTitleMatchScore`, `HasFavicon`, `Robots`, `IsResponsive`, `NoOfURLRedirect`, `NoOfSelfRedirect`, `HasDescription`, `NoOfPopup`, `NoOfiFrame`, `HasExternalFormSubmit`, `HasSocialNet`, `HasSubmitButton`, `HasHiddenFields`, `HasPasswordField`, `Bank`, `Pay`, `Crypto`, `HasCopyrightInfo`, `NoOfImage`, `NoOfCSS`, `NoOfJS`, `NoOfSelfRef`, `NoOfEmptyRef`, `NoOfExternalRef`, `label`.

`URL` is a string; `label` is an integer; FILENAME/Domain/TLD/Title are descriptive strings, remaining supplied columns are lexical/webpage-derived numeric indicators. V1 deliberately reads only URL and label for modeling and never uses webpage-derived fields. The following URL/label values are preserved original columns with raw record lineage (from existing processed metadata).

| Raw source_record | Original URL | Original label | Mapped label |
| --- | --- | --- | --- |
| 2 | hxxps://www[.]southbankmosaics[.]com | 1 | 0 |
| 4 | hxxps://www[.]voicefmradio[.]co[.]uk | 1 | 0 |
| 13 | hxxp://www[.]teramill[.]com | 0 | 1 |

### PhishTank

Original columns: `phish_id`, `url`, `phish_detail_url`, `submission_time`, `verified`, `verification_time`, `online`, `target`. Identifiers are integer-like text; url/phish_detail_url are strings; verified/online are provider flags, submission_time/verification_time are timestamp strings; target is a provider category where present. These three raw prefix records are not promised to survive V1 overlap exclusions.

| phish_id | URL | verified | online | submission_time |
| --- | --- | --- | --- | --- |
| [number redacted] | hxxps://allegrolokalnie[.][number redacted][.]sbs | yes | yes | 2026-11-13T23:33:17+00:00 |
| [number redacted] | hxxps://joyeriasp[.]shop/es/?[query redacted] | yes | yes | 2026-11-13T23:27:59+00:00 |
| [number redacted] | hxxps://podular[.]site/?[query redacted] | yes | yes | 2026-11-13T23:27:59+00:00 |

## Processed datasets and splits

### Email — src/preprocessing/emails.py

Decode MIME parts; extract URL occurrences before cleaning HTML; remove script/style and tags; unescape entities; normalize Unicode NFC and whitespace. Preserve case/punctuation for downstream features. Empty visible bodies/parsing errors are excluded. Exact cleaned body hashes deduplicate messages; long shared paragraphs/quoted threads produce conservative duplicate groups. Conflicting-label groups are excluded, not relabeled. Seeded group-aware greedy splitting targets 70/15/15 with class proportions. This reduces duplicate/thread leakage, but cannot prove semantic independence. CSV labels are parsed literally; Enron source membership gives label 0.

Removed: {'empty_body': 5, 'parsing_error': 12, 'exact_duplicate': 274455, 'conflicting_group': 10}. Saved source class counts: `{'enron': {'0': 243224}, 'Nazario_5': {'0': 1242, '1': 1529}, 'email_text': {'1': 29914, '0': 23743}}`.

| Output under data/ | Rows | Legitimate / phishing |
| --- | --- | --- |
| processed/emails/emails_cleaned.parquet | 299652 | 268209 / 31443 |
| splits/emails/train.parquet | 209758 | 187747 / 22011 |
| splits/emails/validation.parquet | 44947 | 40231 / 4716 |
| splits/emails/test.parquet | 44947 | 40231 / 4716 |

All four outputs share this actual schema (Arrow types):

`record_id: large_string`, `source: large_string`, `source_record: large_string`, `original_label: large_string`, `label: int64`, `label_status: large_string`, `subject: large_string`, `sender: large_string`, `date: large_string`, `headers: large_string`, `text: large_string`, `body: large_string`, `urls: list<element: string>`, `url_count: int64`, `exact_hash: large_string`, `duplicate_group: large_string`, `split: large_string`.

`record_id` identifies the source record; source/source_record/original_label/label_status retain provenance. `text` is cleaned subject + body, `body` cleaned body, `urls` (Arrow list element type string) retains extracted URL metadata, `url_count` counts pre-cleaning occurrences. Sender/date/headers are provenance, never model features. exact_hash/duplicate_group implement deduplication/grouping; split records assigned membership. Cleaning report and work.sqlite are local metadata/workspace, not model inputs.

Real processed training records (also present in cleaned output); validation/test use the same schema and their saved membership is unchanged:

| Source / source_record | label | subject | text excerpt | url_count | split |
| --- | --- | --- | --- | --- | --- |
| enron/allen-p/_sent_mail/1. | 0 |  | Here is our forecast | 0 | train |
| Nazario_5/5 | 0 | FW: ENA Upstream Company information | FW: ENA Upstream Company information [name redacted], We are currently trading under GTC Spot contracts with El Paso for ENA … | 0 | train |
| Nazario_5/1503 | 1 | Verify Your Account | Verify Your Account Business with cPanel & WHM Dear client, Our Technical Services Department are carrying out a planned soft… | 0 | train |
| email_text/2 | 1 |  | do you feel the pressure to perform and not rising to the occasion try v ia gr a your anxiety will be a thing of the past and… | 0 | train |

Before/after for the **same** Nazario_5 record 5 (body excerpts; subject is added separately to model text):

| Original body | Cleaned body | Original → final label |
| --- | --- | --- |
| [name redacted], ⏎  ⏎ We are currently trading under GTC Spot contracts with El Paso for ENA Upstream.  Would you prefer to r… | [name redacted], We are currently trading under GTC Spot contracts with El Paso for ENA Upstream. Would you prefer to review … | 0 → 0 |

### URLs — src/preprocessing/urls.py

Accept only HTTP(S), valid host/port syntax, no whitespace/control characters. Keep url_original intact. The separate normalized URL lowercases scheme/hostname, performs IDNA conversion and removes surrounding whitespace. Canonical keys ignore fragments and default ports and merge empty/root paths; query order and percent escapes remain intact. Exact/canonical duplicates and conflicting-label keys are excluded. No DNS/HTTP queries occur. Existing internal splits use 20 StratifiedGroupKFold hostname folds allocated 14/3/3 with seed 42; registered-domain/temporal disjointness is **not** guaranteed. PhishTank external excludes internal URL keys and hostnames, but was already inspected and must not be called an untouched V3 benchmark.

| Output under data/ | Rows | Legitimate / phishing |
| --- | --- | --- |
| processed/urls/urls_cleaned.parquet | 234687 | 134849 / 99838 |
| splits/urls/train.parquet | 164283 | 94396 / 69887 |
| splits/urls/validation.parquet | 35202 | 20226 / 14976 |
| splits/urls/test.parquet | 35202 | 20227 / 14975 |
| processed/urls/phishtank_external.parquet | 24771 | 0 / 24771 |

Internal schema: `url_original: large_string`, `url: large_string`, `hostname: large_string`, `url_key: large_string`, `has_invalid_percent_escape: bool`, `source: large_string`, `source_record: int64`, `original_label: large_string`, `label: int64`, `phish_id: large_string`, `submission_time: large_string`.

url_original is the raw feature input; url is normalized, hostname parsed, url_key canonical; has_invalid_percent_escape flags syntax; source/source_record/original_label retain provenance; label is normalized; phish_id/submission_time are nullable provider metadata. Only the 13 derived lexical columns reach the classifier.

| source_record | url_original | normalized url | original → mapped label |
| --- | --- | --- | --- |
| 2 | hxxps://www[.]southbankmosaics[.]com | hxxps://www[.]southbankmosaics[.]com | 1 → 0 |
| 4 | hxxps://www[.]voicefmradio[.]co[.]uk | hxxps://www[.]voicefmradio[.]co[.]uk | 1 → 0 |
| 13 | hxxp://www[.]teramill[.]com | hxxp://www[.]teramill[.]com | 0 → 1 |

## Feature datasets and real values

src/features/pipeline.py verifies split/row identity and fits learned transformations on **training only**. No source/label identifiers enter predictors. Metadata and labels remain alongside matrices for alignment checks. No transformer is fitted again for inference.

Email outputs: data/features/emails/{train,validation,test}.npz (float32 CSR) and matching {split}_rows.parquet (record_id,label). Shapes are (209758,50006), (44947,50006), (44947,50006). Word **TF-IDF** downweights corpus-common terms; **unigrams/bigrams** represent individual tokens and adjacent pairs. Fitted vocabulary has 50,000 columns, min_df=2, sublinear_tf=True, L2 text normalization. Six numerical columns are appended in order: text_length, word_count, url_count, suspicious_keyword_count, subject_length, exclamation_count. They measure message/subject size, links, suspicious words and punctuation. MaxAbsScaler fitted on training rescales each numerical column without centering/densifying. New values may exceed one. CSR stores only nonzero values; never convert these matrices to dense arrays. Saved EmailFeatures implements both TF-IDF and scaling in src/features/email_features.py.

Below are **stored** first-three-row feature values from train.npz, sampled directly from its compressed CSR arrays; only five nonzero, non-identifying vocabulary terms are shown per row. Last-six entries are actual scaled numerical values in the order above. These are excerpts, not full vectors.

| Zero-based train row | TF-IDF term=value | Six scaled numerical values |
| --- | --- | --- |
| 0 | is=0.154435 | 1.17841e-05, 3.92218e-05, 0, 0, 0, 0 |
| 1 | to=0.[number redacted] | 2.23897e-05, 6.86382e-05, 0, 0, 0.[number redacted], 0.[number redacted] |
| 2 | to=0.[number redacted], you=0.[number redacted], of=0.[number redacted], the=0.[number redacted], and=0.[number redacted] | 0.[number redacted], 0.[number redacted], 0, 0, 0, 0 |

URL outputs: data/features/urls/{train,validation,test,external}.parquet: 164283 / 35202 / 35202 / 24771 rows respectively; 13 float32 predictors plus url_key and label for alignment. Random Forest consumes the 13 ordered features without scaling. Implemented in src/features/url_features.py.

| Order | Feature | Measurement / possible signal |
| --- | --- | --- |
| 1 | url_length | Original URL character count |
| 2 | hostname_length | Parsed hostname character count |
| 3 | dot_count | Dots anywhere in URL |
| 4 | subdomain_count | max(hostname labels−2,0); zero for IPs; not PSL-aware |
| 5 | digit_count | Unicode digit count in URL |
| 6 | hyphen_count | Hyphen count |
| 7 | special_character_count | Non-alphanumeric character count, including scheme punctuation |
| 8 | is_https | 1 if parsed scheme is HTTPS |
| 9 | hostname_is_ip | 1 if hostname parses as IPv4/IPv6 |
| 10 | url_entropy | Shannon entropy over original URL characters |
| 11 | suspicious_keyword_count | Suspicious keyword occurrences (substring matching) |
| 12 | path_length | Parsed path character count |
| 13 | query_parameter_count | Nonempty ampersand-delimited query components |

Length/entropy/structure may expose obfuscation; HTTPS, punctuation and subdomain counts may also become dataset shortcuts. No single feature proves legitimacy. Definitions/order are frozen to preserve pretrained behavior. Actual stored train feature rows:

| Feature | Row 0 | Row 1 | Row 2 |
| --- | --- | --- | --- |
| url_length | 32 | 30 | 34 |
| hostname_length | 24 | 22 | 26 |
| dot_count | 2 | 3 | 2 |
| subdomain_count | 1 | 2 | 1 |
| digit_count | 0 | 0 | 0 |
| hyphen_count | 0 | 0 | 0 |
| special_character_count | 5 | 6 | 5 |
| is_https | 1 | 1 | 1 |
| hostname_is_ip | 0 | 0 | 0 |
| url_entropy | 3.92923 | 4.16474 | 3.91763 |
| suspicious_keyword_count | 1 | 0 | 0 |
| path_length | 0 | 0 | 0 |
| query_parameter_count | 0 | 0 | 0 |

## Regeneration workflow

From repository root in Python 3.11 with requirements installed, after obtaining the exact raw datasets:

```bash
# Original data must already be placed below this separate root's data/raw_data/.
REPRO_ROOT="$HOME/phishing-ml-reproduction"
python -m src.preprocessing.emails --root "$REPRO_ROOT" --seed 42 --workers 4
python -m src.preprocessing.urls --root "$REPRO_ROOT" --seed 42
python -m src.features.pipeline --root "$REPRO_ROOT" --seed 42
python -m src.training.pipeline --root "$REPRO_ROOT" --seed 42
python -m src.evaluation.pipeline --root "$REPRO_ROOT"
```

These are reproduction commands, **not required for pretrained inference**. Existing completed stages verify and skip rather than silently overwrite; training refuses ambiguous partial runs. Keep shipped models/reports and local datasets intact: reproduce in a separate working copy with separately acquired data, not by deleting markers/artifacts. Historical downloads may differ. CSV lineage/label uncertainties mean exact independent regeneration is currently limited. Cleaning reports, split metadata, feature_report.json, schemas and saved comparison reports document the existing run. Never select models or thresholds using test/external results. Training reproduces the original V1 semantics, not the experimental V2/V3 pipelines.
## Additional real split and external records

These bounded samples come from each named local Parquet, with the same redaction/truncation rules. They document saved rows, not newly evaluated predictions.

### data/splits/emails/validation.parquet

| source_record | label | text | url_count |
| --- | --- | --- | --- |
| allen-p/_sent_mail/102. | 0 | FW: fixed forward or other Collar floor gas price terms ---------------------- Forwarded by Phillip K Allen/HOU/ECT on 10/16/… | 0 |
| allen-p/_sent_mail/103. | 0 | Re: FW: fixed forward or other Collar floor gas price terms Mr. Buckner, For delivered gas behind San Diego, Enron Energy Ser… | 0 |
| allen-p/_sent_mail/109. | 0 | Var, Reporting and Resources Meeting ---------------------- Forwarded by Phillip K Allen/HOU/ECT on 10/04/2000 04:23 PM -----… | 0 |

### data/splits/emails/test.parquet

| source_record | label | text | url_count |
| --- | --- | --- | --- |
| allen-p/_sent_mail/10. | 0 | Re: Traveling to have a business meeting takes the fun out of the trip. Especially if you have to prepare a presentation. I w… | 0 |
| allen-p/_sent_mail/105. | 0 | Consolidated positions: Issues & To Do list ---------------------- Forwarded by Phillip K Allen/HOU/ECT on 10/09/2000 02:16 P… | 0 |
| allen-p/_sent_mail/106. | 0 | Consolidated positions: Issues & To Do list ---------------------- Forwarded by Phillip K Allen/HOU/ECT on 10/09/2000 02:00 P… | 0 |

### data/splits/urls/validation.parquet

| source_record | label | url_original |
| --- | --- | --- |
| 3 | 0 | hxxps://www[.]uni-mainz[.]de |
| 5 | 0 | hxxps://www[.]sfnmjournal[.]com |
| 9 | 0 | hxxps://www[.]nerdscandy[.]com |

### data/splits/urls/test.parquet

| source_record | label | url_original |
| --- | --- | --- |
| 35 | 0 | hxxps://www[.]bikeseoul[.]com |
| 36 | 1 | hxxps://hidok4f8zl[.]firebaseapp[.]com/ |
| 37 | 0 | hxxps://www[.]ibeani[.]co[.]uk |

### data/processed/urls/phishtank_external.parquet

| source_record | label | url_original | submission_time |
| --- | --- | --- | --- |
| 2 | 1 | hxxps://allegrolokalnie[.][number redacted][.]sbs | 2026-11-13T23:33:17+00:00 |
| 3 | 1 | hxxps://joyeriasp[.]shop/es/?[query redacted] | 2026-11-13T23:27:59+00:00 |
| 4 | 1 | hxxps://podular[.]site/?[query redacted] | 2026-11-13T23:27:59+00:00 |
