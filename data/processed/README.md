# Processed V1 datasets — transformations, schemas and real examples

This README is self-contained documentation for the **active Email V1 / URL V1** outputs. All Parquet/SQLite/JSON dataset artifacts remain local and Git-ignored; only this README is tracked. No model was trained/refitted and no pipeline or split was rerun to produce these examples. Counts come from saved cleaning reports/Parquet metadata; samples read bounded existing row groups. Inputs and original source details are summarized here and fully documented in the [raw-data README](../raw_data/README.md).

**Final convention: 0 = legitimate, 1 = phishing.** A label describes dataset annotation, not an independent safety assessment. Addresses/personal headers, secret-looking values and long identifiers are redacted; URLs are defanged and query values suppressed. Truncation is marked. Real rows retain their original field names and typed schemas; examples are transposed for readable wide tables. No listed URL was visited.

## 1. Email V1

### Inputs and implementation

- Enron RFC822/MIME maildir: `data/raw_data/emails/enron/maildir/`, presumed legitimate →0; 517,401 discovered messages.
- Mixed-label CSVs: `data/raw_data/emails/nazario/Nazario_5.csv` (3,065) and `email_text.csv` (53,668). Existing numeric 0/1 labels are retained; phishing-versus-spam semantics and exact release lineage remain unresolved.
- Script: `src/preprocessing/emails.py`. `discover` locates source files; `parse` decodes MIME/header/body representations; `prepare` composes cleaned fields/URL evidence; `clean` normalizes text; `urls` extracts URL occurrences; `keys` constructs duplicate/thread grouping keys; `run` ingests/deduplicates/group-splits/exports; `verify` checks persisted alignment and split/group integrity.

### Actual transformations

1. Decode MIME subject/body and prefer readable text; inspect HTML alternatives for links without rendering/executing them. CSV content fields map to the same subject/body representation; missing optional metadata becomes empty values.
2. Count original URL occurrences **before HTML cleaning**. `clean` removes HTML tags and script/style contents, unescapes entities, normalizes NFC and collapses whitespace. It preserves case/punctuation.
3. Store cleaned body in `body`; cleaned subject plus newline plus raw body becomes `text` after whitespace normalization. Derived `url_count` retains pre-cleaning link occurrences. Raw identifying/header data are provenance only.
4. Remove unusable bodies/parsing failures. Deduplicate exact cleaned-body hashes; group shared long normalized paragraphs/quoted threads conservatively. Remove conflicting-label groups rather than silently changing labels.
5. Greedy class-aware group splitting targets 70/15/15, seed 42. Duplicate groups stay together. No registered-domain or temporal separation is claimed for email.

Saved report: five empty bodies, 12 parsing errors, 274,455 exact duplicates excluded during ingestion, ten records in conflicting groups excluded. Other audit counters use different stages: 274,461 total exact duplicate records, 32,626 extra near-duplicate records after exact deduplication, four conflicting groups. Near-duplicate extras are grouped, not all deleted; do not subtract them as additional removals. Source-filtered missingness/duplicate details are in the raw-data README.

### Output paths, counts and splits

| Output path | Rows | Label 0 / label 1 |
| --- | --- | --- |
| data/processed/emails/emails_cleaned.parquet | 299,652 | 268,209 / 31,443 |
| data/splits/emails/train.parquet | 209,758 | 187,747 / 22,011 |
| data/splits/emails/validation.parquet | 44,947 | 40,231 / 4,716 |
| data/splits/emails/test.parquet | 44,947 | 40,231 / 4,716 |

Cleaned source distributions: Enron 243,224 / 0; Nazario_5 1,242 / 1,529; email_text 23,743 / 29,914 (label 0 / 1). Approximately 89.51% of the combined cleaned records are label 0.

Ancillary local outputs: `data/processed/emails/cleaning_report.json` stores counts/limitations/verification, and `work.sqlite` is an ingestion/grouping workspace, not a separately labeled dataset or an inference dependency.

### Final schema (shared by cleaned and all email splits)

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| `record_id` | large_string | Stable source-record identifier |
| `source` | large_string | Source collection name |
| `source_record` | large_string | Original maildir-relative filename or CSV record number |
| `original_label` | large_string | Original supplied label text or source label |
| `label` | int64 | Final project label (0 legitimate / 1 phishing) |
| `label_status` | large_string | Provenance/confidence category, not a predictive feature |
| `subject` | large_string | Cleaned decoded subject, optional empty string |
| `sender` | large_string | Original sender metadata; not model input |
| `date` | large_string | Supplied date metadata, not necessarily collection time |
| `headers` | large_string | Serialized header metadata, not model input |
| `text` | large_string | Cleaned subject + body used for text features |
| `body` | large_string | Cleaned body |
| `urls` | list&lt;element: string&gt; | Extracted original URL occurrence strings |
| `url_count` | int64 | Pre-cleaning URL occurrence count, repetitions retained |
| `exact_hash` | large_string | Digest for exact-body identity |
| `duplicate_group` | large_string | Assigned conservative thread/near-duplicate group |
| `split` | large_string | Assigned train / validation / test membership |

### Real records: `data/processed/emails/emails_cleaned.parquet`

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `record_id` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `source` | enron | Nazario_5 | Nazario_5 |
| `source_record` | [mailbox redacted]/_sent_mail/1. | 2 | 1502 |
| `original_label` | 0 | 0 | 1 |
| `label` | 0 | 0 | 1 |
| `label_status` | presumed_legitimate | provided_binary_semantics_unverified | provided_binary_semantics_unverified |
| `subject` |  | FW: June 29 -- BNA, Inc. Daily Labor Report | DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA |
| `sender` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `date` | Mon, 14 May 2001 16:39:00 -0700 | Fri, 29 Jun 2001 08:36:09 -0500 | 28 Sep 2017 09:57:25 -0400 |
| `headers` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `text` | Here is our forecast | FW: June 29 -- BNA, Inc. Daily Labor Report User ID: [credential redacted] PW: [credential redacted] -----Original Messag… [truncated] | DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA This text is part of the internal format of y… [truncated] |
| `body` | Here is our forecast | User ID: [credential redacted] PW: [credential redacted] -----Original Message----- From: "BNA Highlights" &lt;[email redact… [truncated] | This text is part of the internal format of your mail folder, and is not a real message. It is … [truncated] |
| `urls` | [] | ['hxxp://web[.]bna[.]com', 'hxxp://pubs[.]bna[.]com/ip/BNA/dlr[.]nsf/id/a0a4j5k3h4_', 'hxxp://p… [truncated] | ['hxxp://id-redirection[.]cu[.]cc/', 'hxxp://sitebuilder[.]vpweb[.]com//em/contacts/unsubscribe… [truncated] |
| `url_count` | 0 | 41 | 122 |
| `exact_hash` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `duplicate_group` | 0 | 49506 | 518886 |
| `split` | train | validation | validation |

### Real records: `data/splits/emails/train.parquet`

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `record_id` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `source` | enron | Nazario_5 | Nazario_5 |
| `source_record` | [mailbox redacted]/_sent_mail/1. | 5 | 1503 |
| `original_label` | 0 | 0 | 1 |
| `label` | 0 | 0 | 1 |
| `label_status` | presumed_legitimate | provided_binary_semantics_unverified | provided_binary_semantics_unverified |
| `subject` |  | FW: ENA Upstream Company information | Verify Your Account |
| `sender` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `date` | Mon, 14 May 2001 16:39:00 -0700 | Fri, 29 Jun 2001 10:35:17 -0500 | Fri, 30 Oct 2015 00:00:48 -0500 |
| `headers` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `text` | Here is our forecast | FW: ENA Upstream Company information [name redacted], We are currently trading under GTC Spot c… [truncated] | Verify Your Account Business with cPanel & WHM Dear client, Our Technical Services Department a… [truncated] |
| `body` | Here is our forecast | [name redacted], We are currently trading under GTC Spot contracts with El Paso for ENA Upstrea… [truncated] | Business with cPanel & WHM Dear client, Our Technical Services Department are carrying out a pl… [truncated] |
| `urls` | [] | [] | [] |
| `url_count` | 0 | 0 | 0 |
| `exact_hash` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `duplicate_group` | 0 | 517389 | 518887 |
| `split` | train | train | train |

### Real records: `data/splits/emails/validation.parquet`

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `record_id` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `source` | enron | Nazario_5 | Nazario_5 |
| `source_record` | [mailbox redacted]/_sent_mail/102. | 2 | 1502 |
| `original_label` | 0 | 0 | 1 |
| `label` | 0 | 0 | 1 |
| `label_status` | presumed_legitimate | provided_binary_semantics_unverified | provided_binary_semantics_unverified |
| `subject` | FW: fixed forward or other Collar floor gas price terms | FW: June 29 -- BNA, Inc. Daily Labor Report | DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA |
| `sender` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `date` | Mon, 16 Oct 2000 06:44:00 -0700 | Fri, 29 Jun 2001 08:36:09 -0500 | 28 Sep 2017 09:57:25 -0400 |
| `headers` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `text` | FW: fixed forward or other Collar floor gas price terms ---------------------- Forwarded by [na… [truncated] | FW: June 29 -- BNA, Inc. Daily Labor Report User ID: [credential redacted] PW: [credential redacted] -----Original Messag… [truncated] | DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA This text is part of the internal format of y… [truncated] |
| `body` | ---------------------- Forwarded by [name redacted]/HOU/ECT on 10/16/2000 01:42 PM ------------… [truncated] | User ID: [credential redacted] PW: [credential redacted] -----Original Message----- From: "BNA Highlights" &lt;[email redact… [truncated] | This text is part of the internal format of your mail folder, and is not a real message. It is … [truncated] |
| `urls` | [] | ['hxxp://web[.]bna[.]com', 'hxxp://pubs[.]bna[.]com/ip/BNA/dlr[.]nsf/id/a0a4j5k3h4_', 'hxxp://p… [truncated] | ['hxxp://id-redirection[.]cu[.]cc/', 'hxxp://sitebuilder[.]vpweb[.]com//em/contacts/unsubscribe… [truncated] |
| `url_count` | 0 | 41 | 122 |
| `exact_hash` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `duplicate_group` | 9 | 49506 | 518886 |
| `split` | validation | validation | validation |

### Real records: `data/splits/emails/test.parquet`

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `record_id` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `source` | enron | Nazario_5 | Nazario_5 |
| `source_record` | [mailbox redacted]/_sent_mail/10. | 23 | 1504 |
| `original_label` | 0 | 0 | 1 |
| `label` | 0 | 0 | 1 |
| `label_status` | presumed_legitimate | provided_binary_semantics_unverified | provided_binary_semantics_unverified |
| `subject` | Re: | FW: July 2 -- BNA, Inc. Daily Labor Report | Helpdesk Mailbox Alert!!! |
| `sender` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `date` | Fri, 04 May 2001 13:51:00 -0700 | Mon, 02 Jul 2001 08:34:40 -0500 | Fri, 30 Oct 2015 06:21:59 -0300 (BRT) |
| `headers` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `text` | Re: Traveling to have a business meeting takes the fun out of the trip. Especially if you have … [truncated] | FW: July 2 -- BNA, Inc. Daily Labor Report User ID: [credential redacted] PW: [credential redacted] -----Original Message… [truncated] | Helpdesk Mailbox Alert!!! Your two incoming mails were placed on pending status due to the rece… [truncated] |
| `body` | Traveling to have a business meeting takes the fun out of the trip. Especially if you have to p… [truncated] | User ID: [credential redacted] PW: [credential redacted] -----Original Message----- From: "BNA Highlights" &lt;[email redact… [truncated] | Your two incoming mails were placed on pending status due to the recent upgrade in our database… [truncated] |
| `urls` | [] | ['hxxp://web[.]bna[.]com', 'hxxp://pubs[.]bna[.]com/ip/BNA/dlr[.]nsf/id/a0a4j7h5j9_', 'hxxp://p… [truncated] | [] |
| `url_count` | 0 | 73 | 0 |
| `exact_hash` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `duplicate_group` | 1 | 49505 | 518888 |
| `split` | test | test | test |

### Same-record before → after

These values come from the exact same Nazario_5 raw CSV records and cleaned rows, matched by `source_record`; no synthetic replacement was used. Original label 0/1 is preserved. Raw HTML/whitespace can be visible in excerpts; cleaning may only collapse whitespace for a particular record.

| Raw CSV record | Original body excerpt | Cleaned body excerpt | Original label → final label | URL count |
| --- | --- | --- | --- | --- |
| 2 | User ID:  [credential redacted] ⏎ PW:        [credential redacted] ⏎  ⏎  ⏎  -----Original Message----- ⏎ From:  "BNA High… [truncated] | User ID: [credential redacted] PW: [credential redacted] -----Original Message----- From: "BNA Highlights" &lt;[email redact… [truncated] | 0 → 0 | 41 |
| 1502 | This text is part of the internal format of your mail folder, and is not a real message.  It is… [truncated] | This text is part of the internal format of your mail folder, and is not a real message. It is … [truncated] | 1 → 1 | 122 |

## 2. URL V1 internal dataset

### Inputs, functions and cleaning

- Input: `data/raw_data/urls/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv`, 235,795 rows. Only original `URL` and `label` are read for modeling, **not** its other lexical/webpage-derived attributes.
- Script: `src/preprocessing/urls.py`; `discover` locates CSV sources, `normalize_url` validates/parses/canonicalizes offline, `ingest` maps labels and deduplicates/removes conflicts, `split` assigns hostname-group folds, `run` exports and applies external exclusions, `verify` checks invariants; `completed` checks existing stage completeness.
- Reject unsupported/missing schemes, invalid hosts/ports and embedded whitespace/control characters. Preserve the exact URL in `url_original`. A separate `url` lowercases scheme/host, performs IDNA conversion and trims surrounding whitespace. Canonical `url_key` ignores fragment/default ports and merges empty/root paths. Query order, percent escapes, case in path/query and credentials are not blindly rewritten. The invalid-percent flag records unsupported escape evidence instead of silently repairing it.
- **Label reversal:** original PhiUSIIL 1 (legitimate) → project 0; original 0 (phishing) → project 1.
- Remove two records at one conflicting canonical key first, then 1,106 extra canonical/exact duplicates; retain 234,687. Deduplication uses canonical identity, not only literal strings.
- `split` creates 20 seeded stratified hostname-group folds, allocating 14/3/3; internal hostnames do not cross splits. Related registered domains/subdomains and collection periods are not guaranteed disjoint. Do not call this registered-domain splitting.

### Output paths/counts

| Output path | Rows | Label 0 / label 1 |
| --- | --- | --- |
| data/processed/urls/urls_cleaned.parquet | 234,687 | 134,849 / 99,838 |
| data/splits/urls/train.parquet | 164,283 | 94,396 / 69,887 |
| data/splits/urls/validation.parquet | 35,202 | 20,226 / 14,976 |
| data/splits/urls/test.parquet | 35,202 | 20,227 / 14,975 |

### Final URL schema

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| `url_original` | large_string | Original unmodified string; input to V1 lexical features |
| `url` | large_string | Separate normalized URL for inspection |
| `hostname` | large_string | Parsed/normalized hostname |
| `url_key` | large_string | Canonical duplicate identity, not a model feature |
| `has_invalid_percent_escape` | bool | Flag for malformed percent-escape syntax |
| `source` | large_string | Source name, provenance only |
| `source_record` | int64 | Original CSV record number, starting at 2 |
| `original_label` | large_string | Original source label text |
| `label` | int64 | Final 0 legitimate / 1 phishing |
| `phish_id` | large_string | Provider identifier; empty for PhiUSIIL |
| `submission_time` | large_string | Provider submission time; empty for PhiUSIIL |

### Real records: `data/processed/urls/urls_cleaned.parquet`

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `url_original` | hxxps://www[.]southbankmosaics[.]com | hxxps://www[.]uni-mainz[.]de | hxxp://www[.]teramill[.]com |
| `url` | hxxps://www[.]southbankmosaics[.]com | hxxps://www[.]uni-mainz[.]de | hxxp://www[.]teramill[.]com |
| `hostname` | www.southbankmosaics.com | www.uni-mainz.de | www.teramill.com |
| `url_key` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `has_invalid_percent_escape` | False | False | False |
| `source` | phiusiil | phiusiil | phiusiil |
| `source_record` | 2 | 3 | 13 |
| `original_label` | 1 | 1 | 0 |
| `label` | 0 | 0 | 1 |
| `phish_id` |  |  |  |
| `submission_time` |  |  |  |

### Real records: `data/splits/urls/train.parquet`

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `url_original` | hxxps://www[.]southbankmosaics[.]com | hxxps://www[.]voicefmradio[.]co[.]uk | hxxp://www[.]teramill[.]com |
| `url` | hxxps://www[.]southbankmosaics[.]com | hxxps://www[.]voicefmradio[.]co[.]uk | hxxp://www[.]teramill[.]com |
| `hostname` | www.southbankmosaics.com | www.voicefmradio.co.uk | www.teramill.com |
| `url_key` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `has_invalid_percent_escape` | False | False | False |
| `source` | phiusiil | phiusiil | phiusiil |
| `source_record` | 2 | 4 | 13 |
| `original_label` | 1 | 1 | 0 |
| `label` | 0 | 0 | 1 |
| `phish_id` |  |  |  |
| `submission_time` |  |  |  |

### Real records: `data/splits/urls/validation.parquet`

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `url_original` | hxxps://www[.]uni-mainz[.]de | hxxps://www[.]sfnmjournal[.]com | hxxp://www[.]ooguy[.]com |
| `url` | hxxps://www[.]uni-mainz[.]de | hxxps://www[.]sfnmjournal[.]com | hxxp://www[.]ooguy[.]com |
| `hostname` | www.uni-mainz.de | www.sfnmjournal.com | www.ooguy.com |
| `url_key` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `has_invalid_percent_escape` | False | False | False |
| `source` | phiusiil | phiusiil | phiusiil |
| `source_record` | 3 | 5 | 39 |
| `original_label` | 1 | 1 | 0 |
| `label` | 0 | 0 | 1 |
| `phish_id` |  |  |  |
| `submission_time` |  |  |  |

### Real records: `data/splits/urls/test.parquet`

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `url_original` | hxxps://www[.]bikeseoul[.]com | hxxps://www[.]ibeani[.]co[.]uk | hxxps://hidok4f8zl[.]firebaseapp[.]com/ |
| `url` | hxxps://www[.]bikeseoul[.]com | hxxps://www[.]ibeani[.]co[.]uk | hxxps://hidok4f8zl[.]firebaseapp[.]com/ |
| `hostname` | www.bikeseoul.com | www.ibeani.co.uk | hidok4f8zl.firebaseapp.com |
| `url_key` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `has_invalid_percent_escape` | False | False | False |
| `source` | phiusiil | phiusiil | phiusiil |
| `source_record` | 35 | 37 | 36 |
| `original_label` | 1 | 1 | 0 |
| `label` | 0 | 0 | 1 |
| `phish_id` |  |  |  |
| `submission_time` |  |  |  |

### Same-record original → processed URL examples

Values below come from raw CSV records 2, 4 and 13 and their preserved original columns in cleaned data. These particular source URLs are already in the normalizer's format: unchanged strings are a valid observed outcome, not an invented normalization example. The label change is explicit.

| Original raw record | Raw URL | Processed url | Original → mapped label |
| --- | --- | --- | --- |
| 2 | hxxps://www[.]southbankmosaics[.]com | hxxps://www[.]southbankmosaics[.]com | 1 → 0 |
| 4 | hxxps://www[.]voicefmradio[.]co[.]uk | hxxps://www[.]voicefmradio[.]co[.]uk | 1 → 0 |
| 13 | hxxp://www[.]teramill[.]com | hxxp://www[.]teramill[.]com | 0 → 1 |

## 3. URL V1 external PhishTank dataset

- Input: `data/raw_data/urls/phishtank/online-valid.csv` (45,341), independent source of provider/community-verified phishing **at snapshot time**.
- Same script and `ingest`/`normalize_url` functions as above; require verified=yes AND online=yes; assign final label 1, original_label=`verified_phishing`.
- 28 syntax-invalid records removed; 137 canonical duplicates removed, retaining 45,176 before primary overlap exclusions. Remove 1,979 primary URL-key overlaps and 18,426 additional primary-hostname overlaps. Final **24,771 phishing-only rows**, zero legitimate rows.
- Output: `data/processed/urls/phishtank_external.parquet`. No train/validation/test subdivision: this is the already-inspected external evaluation snapshot. Do not use it as an untouched V3 benchmark.
- Actual schema matches the URL table above; phish_id/submission_time now contain provider metadata. `submission_time` is stored as text, not parsed timestamp type. No original raw detail-URL/target/verification-time columns are retained because V1 ingestion selects only the five required provider columns.

### Real external processed records

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `url_original` | hxxps://allegrolokalnie[.][number redacted][.]sbs | hxxps://joyeriasp[.]shop/es/?[query redacted] | hxxps://podular[.]site/?[query redacted] |
| `url` | hxxps://allegrolokalnie[.][number redacted][.]sbs | hxxps://joyeriasp[.]shop/es/?[query redacted] | hxxps://podular[.]site/?[query redacted] |
| `hostname` | allegrolokalnie.[number redacted].sbs | joyeriasp.shop | podular.site |
| `url_key` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `has_invalid_percent_escape` | False | False | False |
| `source` | phishtank | phishtank | phishtank |
| `source_record` | 2 | 3 | 4 |
| `original_label` | verified_phishing | verified_phishing | verified_phishing |
| `label` | 1 | 1 | 1 |
| `phish_id` | [number redacted] | [number redacted] | [number redacted] |
| `submission_time` | 2026-11-13T23:33:17+00:00 | 2026-11-13T23:27:59+00:00 | 2026-11-13T23:27:59+00:00 |

## Limitations, preservation and reproduction

Email labels remain source-dependent and partly semantically unverified. URL processing does **not** fix PhiUSIIL source-format bias: the legitimate source class uses HTTPS + www + empty path. Deduplication and split hygiene cannot remove those label shortcuts. Random splits within biased sources and phishing-only external recall do not establish realistic benign FPR. Historical phishing may expire/change; neither raw nor processed labels were reverified online.

`data/processed/urls/cleaning_report.json` records source counts, removals, mapping and split verification. Historical `data/processed/urls_v2/` remains local/ignored as experimental research, outside the active V1 outputs documented here. Feature matrices live separately under `data/features/`, and models never read processed datasets during inference. No original/processed/split file or model was rewritten for this documentation.

For a new reproduction, run from the repository using a **separate root** with the original raw inputs already placed below `data/raw_data/`; do not delete existing outputs or change existing split membership:

```bash
REPRO_ROOT="$HOME/phishing-ml-reproduction"
python -m src.preprocessing.emails --root "$REPRO_ROOT" --seed 42 --workers 4
python -m src.preprocessing.urls --root "$REPRO_ROOT" --seed 42
```

Existing complete stages verify/skip; historical downloads may differ. See [original datasets](../raw_data/README.md), [feature dataset guide](../features/README.md), and [central reproduction documentation](../../docs/datasets.md). This README documents existing outputs; the commands above were **not run** to refresh examples.
