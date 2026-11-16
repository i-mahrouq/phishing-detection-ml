# Historical URL research

Email V1 and URL V1 are the supported pipeline. URL V2/V3 research is archived locally under archive/research (ignored by Git); original experimental models, datasets and reports remain in their original local directories and are ignored. Historical scripts are preserved as reference snapshots, not maintained entry points.

V2 selected character TF-IDF + Linear SVM. At its frozen validation-selected margin, internal test F1 was 0.997558, but Tranco candidate-legitimate FPR was 99.9820%. A later audit reproduced all 1,203,535 feature rows exactly and identified source-format shortcuts: all 134,850 raw PhiUSIIL legitimate URLs used HTTPS, www and no path. These findings also warn against assuming URL V1 generalizes because it uses the same source.

V3 audited sources and qualified bounded samples; no final dataset, splits or models were produced. LegitPhish reused most legacy benign strings, and crawl membership supplied no legitimate safety labels. Reconstruction remained blocked. Do not describe these observed benchmarks as untouched future tests. No experimental results replace the V1 metrics in reports/evaluation.

User-authored notebooks and all data remain local. Archive removal, notebook publication and dataset redistribution require separate review. No archived files were deleted in cleanup.
