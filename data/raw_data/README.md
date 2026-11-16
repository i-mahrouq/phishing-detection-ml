# Original datasets — sources, schemas and real examples

This directory contains **local, Git-ignored source data**. This README is tracked; dataset files and archives are not. No datasets are required to use the pretrained Email V1 / URL V1 predictors. No submitted/listed URL was visited to prepare this documentation.

**Scope:** Enron, Nazario_5, email_text and PhiUSIIL support V1 training; PhishTank supports V1 external evaluation. OpenPhish, Tranco, LegitPhish and Common Crawl remain local historical V2/V3 research material, **not active V1 inputs**. Their presence does not make them suitable training data.

**Evidence:** counts/quality statistics reuse existing cleaning reports and V3 audits. Only the two email CSVs needed a one-pass streaming count of raw labels/missing fields; no full dataset was loaded into memory. Examples are actual bounded records; fields are preserved, long cells truncated (…), personal headers/addresses, mailbox identities and credential-like values redacted, URL schemes/domains defanged and query strings suppressed. Redacted/shortened values are not directly usable training examples. Publisher claims are distinguished from independent verification.

**Source-link status:** exact Kaggle URLs were recovered from Git revision `a2e5f9d` of the original README. Both dataset page titles were checked on 2026-11-16. This confirms the referenced pages exist, not that a particular local file is byte-identical to their current release. No datasets were downloaded. Source usage terms remain those of their publishers.

## Inventory and model usage

| Local source | Raw records | Original labels / evidence | V1 use |
| --- | --- | --- | --- |
| Enron maildir | 517,401 discovered; 517,389 parsed | No phishing labels; membership-inferred legitimate | Email training/internal splits |
| Nazario_5.csv | 3,065 | Supplied numeric labels; semantics unresolved | Email training/internal splits |
| email_text.csv | 53,668 | Supplied numeric labels; semantics unresolved | Email training/internal splits |
| PhiUSIIL CSV | 235,795 | 1=legitimate (134,850); 0=phishing (100,945) | URL training/internal splits |
| PhishTank CSV | 45,341 | verified=yes, online=yes; phishing | External phishing-only check |
| OpenPhish legacy/new feed | 300 / 300 | Source-inferred phishing; two different snapshots | None |
| Tranco top-1m | 1,000,000 | Ranked domains; no verified benign labels | None |
| LegitPhish v2 CSV | 101,219 | 0=phishing (63,678), 1=legitimate (37,540), unknown (1) | None |
| Common Crawl .org/.com index samples | 1,000 / 674 | Unlabeled crawl metadata | None |

## 1. Enron email corpus

Historical corporate email released for email analysis and research, containing communications presumed legitimate by this project. That assumption is **not a verified absence of phishing/spam**.

- **Original provider:** Enron corpus curated/released through CALO/Carnegie Mellon University; official page: [CMU Enron Email Dataset](https://www.cs.cmu.edu/~enron/).
- **Historical project Kaggle reference:** [William Cukierski's Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset). The matching local snapshot/download route is **unverified**. In particular, a Kaggle CSV export must not be confused with this local maildir.
- **Original/local filenames:** `emails/enron/enron_mail_20150507.tar.gz` (retained archive), extracted RFC822 files in `emails/enron/maildir/`. Paths in this README are relative to `data/raw_data/` unless stated otherwise.
- **Format/count:** variable-header RFC822/MIME messages, 517,401 discovered; 517,389 parsed and 12 parsing errors. No original class column. Project assigns all usable records label 0 from source membership.

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| Subject | MIME-decoded string | Subject line; optional |
| From / To / Cc / Bcc | Address/header strings | Correspondents; personal values redacted |
| Date | Header string | Sender-supplied timestamp, not a verified collection time |
| Message-ID / References / In-Reply-To | Header strings | Message/thread metadata where present; omitted/redacted in examples |
| MIME-Version / Content-Type / Content-Transfer-Encoding | Header strings | MIME structure and decoding instructions |
| X-* and other headers | Optional strings | Corpus/mail-client metadata; header set varies by record |
| Body / MIME parts | Decoded text or HTML | Message content; attachments are not executed |

Real message excerpts (a projection of variable MIME headers; `source file` is lineage, not an original message header):

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `source file` | [mailbox redacted]/_sent_mail/1. | [mailbox redacted]/_sent_mail/100. | [mailbox redacted]/_sent_mail/1000. |
| `Subject` |  | Re: test |  |
| `From` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `To` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `Date` | Mon, 14 May 2001 16:39:00 -0700 | Wed, 18 Oct 2000 03:00:00 -0700 | Mon, 23 Oct 2000 06:13:00 -0700 |
| `Content-Type` | text/plain; charset="us-ascii" | text/plain; charset="us-ascii" | text/plain; charset="us-ascii" |
| `Body` | Here is our forecast ⏎  ⏎   | test successful.  way to go!!! | [name redacted], ⏎  ⏎  Can you send me a schedule of the salary and level of everyone in the  ⏎… [truncated] |

**Missingness/limitations:** 19,187 missing subjects, 30 malformed records flagged by the parser; no missing sender/date/body recorded by the saved ingestion report (flags need not mean exclusion). 274,159 source exact duplicates were observed. Historical era, corporate vocabulary, thread repetition, privacy-sensitive content and a single presumed-benign source create bias. The archive duplicates extracted content and is not an additional dataset; its member count was not rescanned.

## 2. Nazario_5.csv (local mixed-label repackaging)

A local CSV of email content and metadata, used as supplied labeled email examples. Its title suggests a phishing corpus, but it also contains class 0 records. The project cannot establish the exact original collection methodology or whether all positive labels mean phishing rather than spam.

- **Historical download attribution:** [Rohan Sood's Phishing email dataset Nazario_5 and TREC07](https://www.kaggle.com/datasets/rohansood98/phishing-email-dataset-nazario-5-and-trec07), recorded in the old project README. Exact member-to-release lineage/redistribution terms remain **unverified**.
- **Original phishing collection context:** [José Nazario's phishing corpus](https://monkey.org/~jose/phishing/). This is an upstream MBOX collection, **not proof that the local mixed-label CSV is an unmodified original Nazario release**. Do not silently equate the two formats or label definitions.
- **Filename/path:** `emails/nazario/Nazario_5.csv`; also retained as a member of `emails/nazario/archive.zip`.
- **Use:** Email V1 ingestion preserves numeric labels literally: supplied 0→project 0; supplied 1→project 1, with uncertain semantic provenance.

**Actual raw records:** 3,065; original class distribution: `0`=1,500, `1`=1,565. Numeric label semantics remain unverified independently of this count.

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| sender / receiver | CSV strings | Correspondent addresses/metadata, redacted |
| date | CSV string | Supplied date; not necessarily acquisition/verification time |
| subject | CSV string | Email subject |
| body | CSV string | Email body, possibly multiline/HTML |
| label | Integer-like CSV field (0/1) | Supplied target; retained literally |
| urls | CSV string | Serialized supplied URL metadata; ingestion extracts URLs from message text itself |

Real original records 2, 3, 1502 (CSV record enumeration starts at 2 after the header; multiline cells are one record):

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `sender` | [personal header redacted] | [personal header redacted] | [personal header redacted] |
| `receiver` | [personal header redacted] | [personal header redacted] |  |
| `date` | Fri, 29 Jun 2001 08:36:09 -0500 | Fri, 29 Jun 2001 09:37:04 -0500 | 28 Sep 2017 09:57:25 -0400 |
| `subject` | FW: June 29 -- BNA, Inc. Daily Labor Report | NGX failover plan. | DON'T DELETE THIS MESSAGE -- FOLDER INTERNAL DATA |
| `body` | User ID:  [credential redacted] ⏎ PW:        [credential redacted] ⏎  ⏎  ⏎  -----Original Message----- ⏎ From:  "BNA High… [truncated] |  ⏎ Hi Chris,   ⏎  ⏎ Tonight we are rolling out a new report.  Currently, only you and Jon have … [truncated] | This text is part of the internal format of your mail folder, and is not a real message.  It is… [truncated] |
| `label` | 0 | 0 | 1 |
| `urls` | ['hxxp://web[.]bna[.]com', 'hxxp://pubs[.]bna[.]com/ip/BNA/dlr[.]nsf/id/a0a4j5k3h4_', 'hxxp://p… [truncated] | [] | 1 |

**Raw empty/whitespace-only cells:** `subject`=50, `receiver`=113, `sender`=2, `date`=3, `body`=2. This is a streamed raw-cell count; saved preprocessing missingness flags can differ after MIME/HTML cleaning.

**Limitations:** saved ingestion observed 292 source exact duplicates. Mixed benign/phishing contents and source/thread overlap (including Enron-like messages) prevent a clean source-independent interpretation. Raw class imbalance, missing provenance, uncertain original dates/usage rights and possible cross-source duplicates limit generalization. The archive is a container of the same two CSVs, not an extra corpus.

## 3. email_text.csv (local binary-label email collection)

Text-only email examples used by Email V1, including apparent spam/phishing and ordinary messages. Original acquisition purpose/creators and exact phishing-vs-spam definition are **unverified**.

- **Exact original dataset page:** unverified. The same [historical Kaggle collection](https://www.kaggle.com/datasets/rohansood98/phishing-email-dataset-nazario-5-and-trec07) is a **candidate attribution**, not a confirmed matching source for this member. No different source is inferred from text/filename alone.
- **Original filename/local path:** `emails/nazario/email_text.csv`; also retained in `emails/nazario/archive.zip`.
- **Format/use:** CSV, numeric labels preserved literally (0→0, 1→1) in Email V1. No subjects/headers/dates are supplied.

**Actual raw records:** 53,668; original class distribution: `0`=23,745, `1`=29,923. Numeric label semantics remain unverified independently of this count.

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| label | Integer-like CSV field (0/1) | Supplied target, not independently verified phishing status |
| text | CSV string | Email content, with no separate subject or header columns |

Real original records 3, 10, 2 (CSV record enumeration starts at 2 after the header; multiline cells are one record):

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `label` | 0 | 0 | 1 |
| `text` | hi i've just updated from the gulus and i check on other mirrors it seems there is a little typ… [truncated] | hi i have to use r to find out the escapenumber confidence interval for the sensitivity and spe… [truncated] | do you feel the pressure to perform and not rising to the occasion try v ia gr a your anxiety w… [truncated] |

**Raw empty/whitespace-only cells:** none. This is a streamed raw-cell count; saved preprocessing missingness flags can differ after MIME/HTML cleaning.

**Limitations:** saved ingestion observed 10 source exact duplicates. The positive sample includes promotional spam: class 1 must not be described as uniformly verified phishing. Raw class imbalance, missing provenance, uncertain original dates/usage rights and possible cross-source duplicates limit generalization. The archive is a container of the same two CSVs, not an extra corpus.

## 4. PhiUSIIL phishing URL dataset

Full legitimate/phishing URL strings with supplied lexical and webpage-derived attributes, published for phishing website detection research.

- **Creators/provider:** Arvind Prasad and Shalini Chandra; UCI Machine Learning Repository. [Exact official dataset page](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset). No Kaggle acquisition is established for this snapshot. License: CC BY 4.0; retain UCI attribution.
- **Original filename/path:** `urls/phiusiil/PhiUSIIL_Phishing_URL_Dataset.csv`.
- **Records/classes:** 235,795; original `label=1` legitimate 134,850; `label=0` phishing 100,945. V1 reverses labels to project convention (0 legitimate / 1 phishing).
- **Use:** URL V1 training/validation/test use URL and mapped label only; original webpage-derived features are excluded to avoid introducing unavailable inference signals. UCI donation date: 2024-03-03; source collection/label dates are not provided per row.

CSV storage is text; the audit deliberately read fields as strings. The table lists semantic types, not a claim that the CSV has intrinsic typed columns. Definitions below summarize supplied attribute names; exact derivation of provider-computed scores/ratios must be obtained from the provider and was not reimplemented or verified here.

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| `FILENAME` | string | Provider record/file identifier |
| `URL` | string | Original URL |
| `URLLength` | integer | Provider URL length |
| `Domain` | string | Provider domain |
| `DomainLength` | integer | Provider domain length |
| `IsDomainIP` | 0/1 flag | Provider domain-is-IP indicator |
| `TLD` | string | Top-level domain |
| `URLSimilarityIndex` | numeric score | Provider URL-similarity score |
| `CharContinuationRate` | numeric rate | Provider character-continuation measure |
| `TLDLegitimateProb` | numeric score | Provider TLD legitimacy probability; not project-calibrated |
| `URLCharProb` | numeric score | Provider character-probability score |
| `TLDLength` | integer | TLD character count |
| `NoOfSubDomain` | integer | Provider subdomain count |
| `HasObfuscation` | 0/1 flag | Provider obfuscation indicator |
| `NoOfObfuscatedChar` | integer | Obfuscated character count |
| `ObfuscationRatio` | numeric ratio | Obfuscation ratio |
| `NoOfLettersInURL` | integer | Letter count |
| `LetterRatioInURL` | numeric ratio | Letter ratio |
| `NoOfDegitsInURL` | integer | Digit count (provider spelling retained) |
| `DegitRatioInURL` | numeric ratio | Digit ratio (provider spelling retained) |
| `NoOfEqualsInURL` | integer | Equals-sign count |
| `NoOfQMarkInURL` | integer | Question-mark count |
| `NoOfAmpersandInURL` | integer | Ampersand count |
| `NoOfOtherSpecialCharsInURL` | integer | Other special-character count |
| `SpacialCharRatioInURL` | numeric ratio | Special-character ratio (provider spelling) |
| `IsHTTPS` | 0/1 flag | HTTPS indicator |
| `LineOfCode` | integer | Webpage code line count |
| `LargestLineLength` | integer | Longest code line |
| `HasTitle` | 0/1 flag | Page title presence |
| `Title` | string | Page title |
| `DomainTitleMatchScore` | numeric score | Domain/title match score |
| `URLTitleMatchScore` | numeric score | URL/title match score |
| `HasFavicon` | 0/1 flag | Favicon presence |
| `Robots` | 0/1 flag | Provider robots indicator |
| `IsResponsive` | 0/1 flag | Provider responsive-page indicator |
| `NoOfURLRedirect` | integer | URL redirects |
| `NoOfSelfRedirect` | integer | Self redirects |
| `HasDescription` | 0/1 flag | Description presence |
| `NoOfPopup` | integer | Popup count |
| `NoOfiFrame` | integer | Iframe count |
| `HasExternalFormSubmit` | 0/1 flag | External form submission indicator |
| `HasSocialNet` | 0/1 flag | Social-network links indicator |
| `HasSubmitButton` | 0/1 flag | Submit button presence |
| `HasHiddenFields` | 0/1 flag | Hidden field presence |
| `HasPasswordField` | 0/1 flag | Password field presence |
| `Bank` | 0/1 flag | Provider banking-related indicator |
| `Pay` | 0/1 flag | Provider payment-related indicator |
| `Crypto` | 0/1 flag | Provider cryptocurrency-related indicator |
| `HasCopyrightInfo` | 0/1 flag | Copyright text indicator |
| `NoOfImage` | integer | Image count |
| `NoOfCSS` | integer | CSS count |
| `NoOfJS` | integer | JavaScript count |
| `NoOfSelfRef` | integer | Self-reference count |
| `NoOfEmptyRef` | integer | Empty-reference count |
| `NoOfExternalRef` | integer | External-reference count |
| `label` | integer 0/1 | Original 1 legitimate / 0 phishing |

Actual raw records 2, 4 and 13, with **every original column preserved** (transposed for readability):

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `FILENAME` | 521848.txt | 597387.txt | mw42508.txt |
| `URL` | hxxps://www[.]southbankmosaics[.]com | hxxps://www[.]voicefmradio[.]co[.]uk | hxxp://www[.]teramill[.]com |
| `URLLength` | 31 | 29 | 22 |
| `Domain` | www[.]southbankmosaics[.]com | www[.]voicefmradio[.]co[.]uk | www[.]teramill[.]com |
| `DomainLength` | 24 | 22 | 16 |
| `IsDomainIP` | 0 | 0 | 0 |
| `TLD` | com | uk | com |
| `URLSimilarityIndex` | 100 | 100 | 82.[number redacted] |
| `CharContinuationRate` | 1 | 0.[number redacted] | 1 |
| `TLDLegitimateProb` | 0.[number redacted] | 0.028555 | 0.[number redacted] |
| `URLCharProb` | 0.[number redacted] | 0.[number redacted] | 0.[number redacted] |
| `TLDLength` | 3 | 2 | 3 |
| `NoOfSubDomain` | 1 | 2 | 1 |
| `HasObfuscation` | 0 | 0 | 0 |
| `NoOfObfuscatedChar` | 0 | 0 | 0 |
| `ObfuscationRatio` | 0 | 0 | 0 |
| `NoOfLettersInURL` | 18 | 15 | 10 |
| `LetterRatioInURL` | 0.581 | 0.517 | 0.455 |
| `NoOfDegitsInURL` | 0 | 0 | 0 |
| `DegitRatioInURL` | 0 | 0 | 0 |
| `NoOfEqualsInURL` | 0 | 0 | 0 |
| `NoOfQMarkInURL` | 0 | 0 | 0 |
| `NoOfAmpersandInURL` | 0 | 0 | 0 |
| `NoOfOtherSpecialCharsInURL` | 1 | 2 | 1 |
| `SpacialCharRatioInURL` | 0.032 | 0.069 | 0.045 |
| `IsHTTPS` | 1 | 1 | 0 |
| `LineOfCode` | 558 | 467 | 2 |
| `LargestLineLength` | 9381 | 682 | 87 |
| `HasTitle` | 1 | 1 | 0 |
| `Title` | à¸‚à¹ˆà¸²à¸§à¸ªà¸” à¸‚à¹ˆà¸²à¸§à¸§à¸±à¸™à¸™à¸µà¹‰ à¸‚à¹ˆà¸²à¸§à¸à¸µà¸¬à¸² à¸‚à¹ˆà¸²à¸§à¸šà¸±à¸… [truncated] | voice fm southampton | 0 |
| `DomainTitleMatchScore` | 0 | 46.[number redacted] | 0 |
| `URLTitleMatchScore` | 0 | 46.[number redacted] | 0 |
| `HasFavicon` | 0 | 0 | 0 |
| `Robots` | 1 | 1 | 0 |
| `IsResponsive` | 1 | 1 | 0 |
| `NoOfURLRedirect` | 0 | 0 | 0 |
| `NoOfSelfRedirect` | 0 | 0 | 0 |
| `HasDescription` | 0 | 1 | 0 |
| `NoOfPopup` | 0 | 0 | 0 |
| `NoOfiFrame` | 1 | 0 | 0 |
| `HasExternalFormSubmit` | 0 | 0 | 0 |
| `HasSocialNet` | 0 | 0 | 0 |
| `HasSubmitButton` | 1 | 1 | 0 |
| `HasHiddenFields` | 1 | 1 | 0 |
| `HasPasswordField` | 0 | 0 | 0 |
| `Bank` | 1 | 0 | 0 |
| `Pay` | 0 | 0 | 0 |
| `Crypto` | 0 | 0 | 0 |
| `HasCopyrightInfo` | 1 | 1 | 0 |
| `NoOfImage` | 34 | 10 | 0 |
| `NoOfCSS` | 20 | 2 | 0 |
| `NoOfJS` | 28 | 7 | 0 |
| `NoOfSelfRef` | 119 | 42 | 0 |
| `NoOfEmptyRef` | 0 | 2 | 0 |
| `NoOfExternalRef` | 124 | 5 | 1 |
| `label` | 1 | 1 | 0 |

**Verified limitations:** all 134,850 raw legitimate URLs are HTTPS, www-prefixed and have empty paths (no trailing slash). These formatting shortcuts strongly correlate with the target and do not represent everyday legitimate URLs. The saved audit found 425 literal duplicate records; canonical identities have 1,107 additional duplicates, including one conflicting canonical key (two records). V1 removes conflicts first and then 1,106 duplicates, retaining 234,687. All URLs passed the existing syntax parser, but that neither proves live status nor label correctness. UCI reports no missing values; missingness of unused local webpage-derived columns was not independently recounted. Registered-domain overlap and provider-computed features can leak source/domain identity; only frozen 13 lexical V1 features are active.

## 5. PhishTank verified-online snapshot

A community/provider phishing feed collected for threat intelligence. This project uses a filtered, overlap-excluded subset only for external phishing detection recall.

- **Provider:** PhishTank community/service. [Exact official developer page](https://www.phishtank.org/developer_info.php) and [verification FAQ](https://phishtank.org/faq.php). Historical README references the developer page; the exact original download URL/retrieval timestamp of this local snapshot is unverified. No Kaggle source established.
- **Filename/path:** `urls/phishtank/online-valid.csv`; CSV; 45,341 records, all saved audit records flagged verified/online. No numeric original class; V1 retains verified=yes AND online=yes and assigns label 1.

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| `phish_id` | integer-like text | Provider phishing record identifier |
| `url` | string | Reported phishing URL |
| `phish_detail_url` | string | Provider details-page URL, not fetched |
| `submission_time` | timestamp string | Provider submission time |
| `verified` | yes/no string | Provider/community verification flag |
| `verification_time` | timestamp string | Provider verification time |
| `online` | yes/no string | Online-at-snapshot flag |
| `target` | string | Provider target category |

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `phish_id` | [number redacted] | [number redacted] | [number redacted] |
| `url` | hxxps://allegrolokalnie[.][number redacted][.]sbs | hxxps://joyeriasp[.]shop/es/?[query redacted] | hxxps://podular[.]site/?[query redacted] |
| `phish_detail_url` | hxxp://www[.]phishtank[.]com/phish_detail[.]php?[query redacted] | hxxp://www[.]phishtank[.]com/phish_detail[.]php?[query redacted] | hxxp://www[.]phishtank[.]com/phish_detail[.]php?[query redacted] |
| `submission_time` | 2026-11-13T23:33:17+00:00 | 2026-11-13T23:27:59+00:00 | 2026-11-13T23:27:59+00:00 |
| `verified` | yes | yes | yes |
| `verification_time` | 2026-11-14T14:11:56+00:00 | 2026-11-14T14:11:56+00:00 | 2026-11-14T14:11:56+00:00 |
| `online` | yes | yes | yes |
| `target` | Allegro | Other | Other |

**Quality/limitations:** 28 inputs rejected by the offline parser (21 hostname, seven whitespace/control errors); 137 canonical duplicates removed. After excluding 1,979 internal URL overlaps and 18,426 additional hostname overlaps, V1 exports 24,771 phishing records. Provider-verified is not independent verification by this project; URLs can expire/change. A phishing-only set cannot estimate benign FPR, precision or overall deployment accuracy. Submission/verification times are provider events, not present-day validity evidence.

## 6. OpenPhish community feeds — retained experiments only

Provider threat-intelligence URL lines collected to observe phishing detection outside the internal source. No record has an explicit class column or independent local phishing adjudication; project label 1 was inferred from feed membership.

- **Provider/page:** OpenPhish, [Community and commercial feeds](https://www.openphish.com/phishing_feeds.html); [usage terms](https://www.openphish.com/terms.html).
- **Legacy snapshot:** `urls/openPhis/urls.txt`, 300 URL lines; exact original retrieval endpoint/date unverified, despite the provider attribution.
- **New snapshot:** `urls_v3/openphish/community_feed.txt`, 300 URL lines; official GitHub endpoint recorded below. They are separate collections, not 600 proven unique URLs. Neither is V1 training input.
- **Format/schema:** UTF-8 text; one original URL string per nonempty line, no header, no label/collection-time columns. Labels are source-inferred; acquisition time is snapshot-level only.

Recorded retrieval: `2026-11-16T15:03:30.154542+00:00`. Exact acquisition endpoint: [https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt](https://raw.githubusercontent.com/openphish/public_feed/refs/heads/main/feed.txt). Saved SHA-256: `2e59abd7d633e1df55953e290b409e0586d41f997b5f5d13c81001a95a577e4a`.

Actual first three records in `urls/openPhis/urls.txt`. Provider-restricted URL content is redacted, rather than publishing feed addresses; schemes and string lengths describe the actual records:

| Original line | URL string (redacted) | Original string length |
| --- | --- | --- |
| 1 | hxxp://[host/path/query redacted] | 30 |
| 2 | hxxps://[host/path/query redacted] | 51 |
| 3 | hxxp://[host/path/query redacted] | 46 |

Actual first three records in `urls_v3/openphish/community_feed.txt`. Provider-restricted URL content is redacted, rather than publishing feed addresses; schemes and string lengths describe the actual records:

| Original line | URL string (redacted) | Original string length |
| --- | --- | --- |
| 1 | hxxps://[host/path/query redacted] | 76 |
| 2 | hxxp://[host/path/query redacted] | 35 |
| 3 | hxxps://[host/path/query redacted] | 56 |

**Limitations:** legacy audit: 300 syntax-valid URLs, no canonical duplicates, 282 hosts / 270 registered domains. New audit: 299 syntax-valid of 300, one rejected, no literal duplicates. No reliable per-URL label dates; feed membership is not independent current verification. Usage/redistribution restrictions apply; keep original feeds local. Previously inspected benchmarks are not untouched future holdouts.

## 7. Tranco top-one-million domains — provisional, not verified benign

Domain popularity/ranking research data. The project previously used popularity as candidate-legitimate benchmark membership; **popularity does not prove legitimacy**, and domain names are not real full URL observations.

- **Provider:** Tranco research ranking; [official list service](https://tranco-list.eu/). Snapshot identifier `947V2` is recoverable from the local archive filename, but its exact list-specific page/download URL and retrieval date have not been verified and are not invented here.
- **Files:** `urls/tranco/tranco_947V2-1m.csv.zip` and extracted `urls/tranco/top-1m.csv` (same collection, not two datasets).
- **Format/count:** headerless two-column CSV: position 1 rank (integer), position 2 domain (string); 1,000,000 rows. No original label. Historical label 0 was provisional/source-inferred, not a legitimate ground truth. No V1 use.

| Original column 1: rank | Original column 2: domain |
| --- | --- |
| 1 | google[.]com |
| 2 | cloudflare[.]com |
| 3 | amazonaws[.]com |

**Limitations:** saved audit: 999,994 parser-accepted domain inputs, six rejected, 898 accepted without a recognized registered domain; no duplicate accepted literal/hostname identities. Conversion to `https://domain/` in historical experiments was synthetic formatting, not observed full-URL evidence. No paths, schemes, phishing labels or per-domain verification dates are supplied. Uniform root-only generated URLs create a major distribution shift. Full missingness/ZIP member contents were not rescanned; do not infer safety from rank.

## 8. LegitPhish v2 — quarantined research candidate

A provider-labeled full-URL collection with supplied numerical features, intended for phishing classification research; acquired to assess whether it could improve legitimate URL diversity. **It was not accepted into V1 or final V3 splits.**

- **Creators:** Rachana Potpelwar, Uday Kulkarni and Jaishri Waghmare; Mendeley Data. [Exact dataset page, version 2](https://data.mendeley.com/datasets/hx4m73v2sf/2), DOI `10.17632/hx4m73v2sf.2`, published 2025-05-22; CC BY 4.0. Creator/feature descriptions are provider assertions, not independently validated formulas.
- **Original filename/path:** `urls_v3/legitphish/url_features_extracted1.csv`; CSV, 101,219 rows. Original `ClassLabel=0` phishing 63,678; `ClassLabel=1` legitimate 37,540; **one unknown/missing class**, preserved. Project audit reverses known labels; unknown is never silently assigned.

Recorded retrieval: `2026-11-16T15:05:41.089925+00:00`. Exact acquisition endpoint: [https://data.mendeley.com/public-files/datasets/hx4m73v2sf/files/0ed51e4d-d160-4047-b650-3f0801eea4aa/file_downloaded](https://data.mendeley.com/public-files/datasets/hx4m73v2sf/files/0ed51e4d-d160-4047-b650-3f0801eea4aa/file_downloaded). Saved SHA-256: `8685e7901c3b4e73eb43c106cb166cc932dd14f76960a032c713c2c32f09dbcd`.

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| `URL` | string | Full original URL |
| `url_length` | integer | Provider URL character count |
| `has_ip_address` | 0/1 flag | IP-address indicator |
| `dot_count` | integer | Dot count |
| `https_flag` | 0/1 flag | HTTPS indicator |
| `url_entropy` | float | Provider character entropy |
| `token_count` | integer | Provider token count |
| `subdomain_count` | integer | Provider subdomain count |
| `query_param_count` | integer | Query parameter count |
| `tld_length` | integer | TLD length |
| `path_length` | integer | Path length |
| `has_hyphen_in_domain` | 0/1 flag | Hyphen in domain |
| `number_of_digits` | integer | Digit count |
| `tld_popularity` | 0/1 flag | Provider popular-TLD indicator |
| `suspicious_file_extension` | 0/1 flag | Provider suspicious-extension indicator |
| `domain_name_length` | integer | Domain name length |
| `percentage_numeric_chars` | float | Provider numeric-character percentage |
| `ClassLabel` | 0/1 or missing | Original 1 legitimate / 0 phishing |

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `URL` | hxxps://keraekken-loagginnusa[.]godaddysites[.]com/ | hxxps://metamsk01lgiix[.]godaddysites[.]com/ | hxxps://stackoverflow[.]com/questions/tagged/php |
| `url_length` | 47 | 40 | 46 |
| `has_ip_address` | 0 | 0 | 0 |
| `dot_count` | 2 | 2 | 1 |
| `https_flag` | 1 | 1 | 1 |
| `url_entropy` | 4.[number redacted] | 4.[number redacted] | 4.[number redacted] |
| `token_count` | 6 | 6 | 7 |
| `subdomain_count` | 1 | 1 | 0 |
| `query_param_count` | 1 | 1 | 1 |
| `tld_length` | 3 | 3 | 3 |
| `path_length` | 1 | 1 | 21 |
| `has_hyphen_in_domain` | 1 | 0 | 0 |
| `number_of_digits` | 0 | 2 | 0 |
| `tld_popularity` | 1 | 1 | 1 |
| `suspicious_file_extension` | 0 | 0 | 0 |
| `domain_name_length` | 12 | 12 | 13 |
| `percentage_numeric_chars` | 0 | 5 | 0 |
| `ClassLabel` | 0 | 0 | 1 |

**Verified issues:** 1,323 malformed/unsupported URLs, 346 additional exact duplicate records, 1,019 canonical duplicates; one unknown label. Provider-labeled legitimate rows are 99.9867% HTTPS, 99.0863% www and only 1.4172% non-root paths. **36,935 / 37,540 legitimate rows (98.39%) exactly match raw PhiUSIIL strings**, so this is not an independent new benign source. Provider descriptions claim manual verification but no per-record evidence was acquired; URLhaus-related sourcing raises phishing-versus-malware semantics questions. 49,057 syntax-valid rows lacked a recognized registered domain; deeper breakdown was deferred. No class-dependent normalization or relabeling was performed.

## 9. Common Crawl index samples — unlabeled metadata

Public web-crawl index records collected to assess real full-URL formatting diversity. Crawl inclusion/HTTP 200 is **not a legitimate label**. No pages, WARC payloads or linked URLs were fetched by this project.

- **Provider:** Common Crawl Foundation; [CDXJ index documentation](https://www.commoncrawl.org/cdxj-index), [exact collection API](https://index.commoncrawl.org/CC-MAIN-2026-43-index), [terms](https://www.commoncrawl.org/terms-of-use).
- **Files:** `urls_v3/commoncrawl/CC-MAIN-2026-43_org_sample.jsonl` (1,000 records); `..._com_sample.jsonl` (674). Original retained filenames include the stratum and collection ID.
- **Format:** JSON Lines (one index JSON object per row); fields below are strings, including numeric-looking values. Optional fields can differ by record; empty-string values remain empty.
- **Collection:** CC-MAIN-2026-43 spans 2026-10-05 through 2026-10-18 according to retained metadata. Sampling used bounded first response records for `*.org/*` and `*.com/*`, status 200, not a random web sample. All 1,674 labels remain unknown.

### org sample

Recorded retrieval: `2026-11-16T15:04:01.333953+00:00`. Exact acquisition endpoint: [https://index.commoncrawl.org/CC-MAIN-2026-43-index?url=%2A.org%2F%2A&output=json&page=0&pageSize=1&limit=1000&filter=status%3A200](https://index.commoncrawl.org/CC-MAIN-2026-43-index?url=%2A.org%2F%2A&output=json&page=0&pageSize=1&limit=1000&filter=status%3A200). Saved SHA-256: `1f493f2bb9f8e4eb6b600e23765ddbd6a63abdb1e96275726b6be9910f00dcc2`.

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| `urlkey` | JSON string | SURT/canonical index search key, not a project feature |
| `timestamp` | JSON string | Crawl timestamp YYYYMMDDhhmmss |
| `url` | JSON string | Observed full URL |
| `mime` | JSON string | Reported MIME type |
| `mime-detected` | JSON string | Detected MIME type |
| `status` | JSON string | HTTP response status at crawl time |
| `digest` | JSON string | Provider payload digest |
| `length` | JSON string | WARC compressed record length |
| `offset` | JSON string | WARC byte offset |
| `filename` | JSON string | WARC storage object path; not downloaded |
| `languages` | JSON string | Detected language codes where available |
| `recordid` | JSON string | Provider WARC record identifier |
| `encoding` | JSON string | Detected content encoding where available |

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `urlkey` | org,0--0)/ | org,0--0)/2003/06 | org,0--0)/2011/02 |
| `timestamp` | [number redacted] | [number redacted] | [number redacted] |
| `url` | hxxps://0--0[.]org/ | hxxps://0--0[.]org/2003/06/ | hxxps://0--0[.]org/2011/02/ |
| `mime` | text/html | text/html | text/html |
| `mime-detected` | text/html | text/html | text/html |
| `status` | 200 | 200 | 200 |
| `digest` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `length` | 12973 | 6404 | 9414 |
| `offset` | [number redacted] | [number redacted] | [number redacted] |
| `filename` | crawl-data/CC-MAIN-2026-43/segments/[number redacted].35/warc/[long identifier redacted].warc.g… [truncated] | crawl-data/CC-MAIN-2026-43/segments/[number redacted].35/warc/[long identifier redacted].warc.g… [truncated] | crawl-data/CC-MAIN-2026-43/segments/[number redacted].35/warc/[long identifier redacted].warc.g… [truncated] |
| `languages` | eng | eng | eng |
| `recordid` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `encoding` | UTF-8 | UTF-8 | UTF-8 |

### com sample

Recorded retrieval: `2026-11-16T15:04:08.283159+00:00`. Exact acquisition endpoint: [https://index.commoncrawl.org/CC-MAIN-2026-43-index?url=%2A.com%2F%2A&output=json&page=0&pageSize=1&limit=1000&filter=status%3A200](https://index.commoncrawl.org/CC-MAIN-2026-43-index?url=%2A.com%2F%2A&output=json&page=0&pageSize=1&limit=1000&filter=status%3A200). Saved SHA-256: `5c5794259f2211cbd1c251d51d272a20e1c70499f7ee9f83b4d408561c71ff92`.

| Field(s) | Stored / parsed type | Meaning |
| --- | --- | --- |
| `urlkey` | JSON string | SURT/canonical index search key, not a project feature |
| `timestamp` | JSON string | Crawl timestamp YYYYMMDDhhmmss |
| `url` | JSON string | Observed full URL |
| `mime` | JSON string | Reported MIME type |
| `mime-detected` | JSON string | Detected MIME type |
| `status` | JSON string | HTTP response status at crawl time |
| `digest` | JSON string | Provider payload digest |
| `length` | JSON string | WARC compressed record length |
| `offset` | JSON string | WARC byte offset |
| `filename` | JSON string | WARC storage object path; not downloaded |
| `languages` | JSON string | Detected language codes where available |
| `recordid` | JSON string | Provider WARC record identifier |
| `encoding` | JSON string | Detected content encoding where available |

| Original field | Record 1 | Record 2 | Record 3 |
| --- | --- | --- | --- |
| `urlkey` | com,[long identifier redacted])/ | com,[long identifier redacted])/ | com,[long identifier redacted])/robots.txt |
| `timestamp` | [number redacted] | [number redacted] | [number redacted] |
| `url` | hxxp://[long identifier redacted][.]com/ | hxxps://[long identifier redacted][.]com/ | hxxps://[long identifier redacted][.]com/robots[.]txt |
| `mime` | text/html | text/html | text/plain |
| `mime-detected` | text/html | text/html | text/plain |
| `status` | 200 | 200 | 200 |
| `digest` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `length` | 125667 | 1353 | 1571 |
| `offset` | 909 | [number redacted] | 284762 |
| `filename` | crawl-data/CC-MAIN-2026-43/segments/[number redacted].77/warc/[long identifier redacted].warc.g… [truncated] | crawl-data/CC-MAIN-2026-43/segments/[number redacted].42/warc/[long identifier redacted].warc.g… [truncated] | crawl-data/CC-MAIN-2026-43/segments/[number redacted].42/robotstxt/[long identifier redacted].w… [truncated] |
| `languages` | eng | eng | [missing] |
| `recordid` | [long identifier redacted] | [long identifier redacted] | [long identifier redacted] |
| `encoding` | UTF-8 | UTF-8 | [missing] |

**Limitations:** saved audit found ten canonical duplicate records in .com and four in .org. First-page/domain-stratum sampling, shared hosts and crawl timing introduce selection bias; no reliable phishing/benign verification exists. .org response was intentionally stopped at the 1,000-record bound. Missingness of optional fields was not exhaustively recomputed. Do not invent labels or treat these samples as vetted negatives.

## Containers, metadata and preservation

- `emails/nazario/archive.zip`, Enron TAR and Tranco ZIP are retained original containers; their extracted data are documented above, not double-counted.
- `urls/phiusiil/.ipynb_checkpoints/` retains a duplicate notebook checkpoint CSV; it is not an independently collected dataset and is ignored during discovery.
- `urls_v3/metadata/` contains saved source pages/terms, index collection metadata, the Mendeley file listing and a site JavaScript bundle inspected as text. These are acquisition evidence, **not labeled datasets**; JavaScript was not executed. They remain local and ignored.
- Local `reports/url_v3/dataset_audit/` and `reports/url_v3/source_qualification/` supplied historical counts/bias/overlap evidence. They are archived research outputs, not required for inference or a claim of new V3 work.

For processed schema, transformations and actual before/after rows see the self-contained [processed-data README](../processed/README.md). [Central dataset guide](../../docs/datasets.md) explains features and reproduction. No raw/processed data were changed or deleted while writing this document.
