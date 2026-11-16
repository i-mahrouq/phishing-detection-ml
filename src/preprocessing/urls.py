"""Offline URL preparation. No URLs are fetched and no learned features are fitted."""
from __future__ import annotations
import argparse
from collections import Counter
import ipaddress
import json
import logging
from pathlib import Path
import re
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .emails import digest  # Existing stable content-hash utility; does not run emails.

VERSION = 1
NAMES = ('train', 'validation', 'test')


def normalize_url(value):
    """Conservative normalization; preserve original URL and all lexical evidence.

    Only scheme/hostname case, IDNA host encoding, and surrounding whitespace are
    normalized in model text. Deduplication additionally ignores fragments,
    default ports and the empty-vs-root path. Never decode escapes, sort queries,
    drop credentials, remove tracking fields, or collapse path components.
    """
    raw = '' if value is None else str(value)
    value = raw.strip()
    if not value:
        raise ValueError('empty_url')
    if re.search(r'[\s\x00-\x1f\x7f]', value):
        raise ValueError('embedded_whitespace_or_control')
    try:
        parsed = urlsplit(value)
        if parsed.scheme.lower() not in ('http', 'https'):
            raise ValueError('unsupported_or_missing_scheme')
        host = parsed.hostname
        if not host:
            raise ValueError('missing_hostname')
        port = parsed.port  # Validate numeric range without network access.
        host = host.rstrip('.')
        try:
            host = str(ipaddress.ip_address(host))
        except ValueError:
            host = host.encode('idna').decode('ascii').lower()
            if len(host) > 253 or '.' not in host or any(
                not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label)
                for label in host.split('.')
            ):
                raise ValueError('invalid_hostname')
        hostname = host
        display_host = '[' + host + ']' if ':' in host else host
        # Preserve the original userinfo and explicit port spelling as evidence.
        userinfo = parsed.netloc.rsplit('@', 1)[0] + '@' if '@' in parsed.netloc else ''
        authority_tail = parsed.netloc.rsplit('@', 1)[-1]
        port_text = authority_tail[authority_tail.rfind(':'):] if port is not None else ''
        netloc = userinfo + display_host + port_text
        # Replace authority only; preserve empty query/fragment delimiters too.
        suffix = value[value.index('://') + 3 + len(parsed.netloc):]
        normalized = parsed.scheme.lower() + '://' + netloc + suffix
        default_port = (parsed.scheme.lower(), port) in (('http', 80), ('https', 443))
        identity_netloc = userinfo + display_host + ('' if default_port else (':' + str(port) if port is not None else ''))
        identity = urlunsplit((parsed.scheme.lower(), identity_netloc,
                               parsed.path or '/', parsed.query, ''))
        return dict(url_original=raw, url=normalized, hostname=hostname,
                    url_key=digest(identity),
                    has_invalid_percent_escape=bool(re.search(r'%(?![0-9a-fA-F]{2})', value)))
    except (UnicodeError, ValueError) as exc:
        reason = str(exc)
        if reason not in {'unsupported_or_missing_scheme', 'missing_hostname', 'invalid_hostname'}:
            reason = 'malformed_url_or_port'
        raise ValueError(reason) from exc


def discover(root):
    """Discover by schema, excluding notebook checkpoint copies."""
    found = {'phiusiil': [], 'phishtank': []}
    for path in sorted((Path(root) / 'data/raw_data/urls').rglob('*.csv')):
        if '.ipynb_checkpoints' in path.parts or any(p in path.parts for p in ('processed', 'splits')):
            continue
        columns = set(pd.read_csv(path, nrows=0).columns)
        if {'URL', 'label', 'URLLength', 'Domain'}.issubset(columns):
            found['phiusiil'].append(path)
        elif {'url', 'phish_id', 'verified', 'online'}.issubset(columns):
            found['phishtank'].append(path)
    for source, paths in found.items():
        if len(paths) != 1:
            raise ValueError(f'Expected one {source} CSV, found {len(paths)}: {paths}')
    return {source: paths[0] for source, paths in found.items()}


def ingest(path, source):
    columns = ['URL', 'label'] if source == 'phiusiil' else ['url', 'phish_id', 'verified', 'online', 'submission_time']
    counts = Counter(raw=0, valid=0)
    reasons = Counter()
    records = []
    for chunk in pd.read_csv(path, usecols=columns, dtype=str, keep_default_na=False, chunksize=50000):
        for row in chunk.to_dict('records'):
            row_number = counts['raw'] + 2
            counts['raw'] += 1
            if source == 'phiusiil' and row['label'] not in ('0', '1'):
                reasons['invalid_label'] += 1
                continue
            if source == 'phishtank' and (row['verified'].lower() != 'yes' or row['online'].lower() != 'yes'):
                reasons['not_verified_online'] += 1
                continue
            try:
                record = normalize_url(row['URL' if source == 'phiusiil' else 'url'])
            except ValueError as exc:
                reasons[str(exc)] += 1
                continue
            record.update(source=source, source_record=row_number,
                          original_label=row.get('label', 'verified_phishing'),
                          label=1-int(row['label']) if source == 'phiusiil' else 1,
                          phish_id=row.get('phish_id', ''), submission_time=row.get('submission_time', ''))
            records.append(record)
            counts['valid'] += 1
        logging.info('%s: read %s rows', source, counts['raw'])
    if not records:
        raise ValueError(f'No usable URLs in {path}')
    frame = pd.DataFrame(records)
    conflicts = frame.groupby('url_key').label.nunique().gt(1)
    conflict_keys = set(conflicts[conflicts].index)
    bad = frame.url_key.isin(conflict_keys)
    counts['conflicting_keys'] = len(conflict_keys)
    reasons['conflicting_label_records'] = int(bad.sum())
    frame = frame.loc[~bad].copy()
    duplicate = frame.duplicated('url_key')
    reasons['exact_or_canonical_duplicate'] = int(duplicate.sum())
    frame = frame.loc[~duplicate].reset_index(drop=True)
    counts['retained'] = len(frame)
    return frame, dict(counts=counts, removed=reasons, path=str(path))


def split(frame, seed):
    """Twenty stratified hostname-group folds allocated 14/3/3 (70/15/15).

    Hostname isolation is stronger than exact URL isolation. Related subdomains
    may still overlap; these are not registrable-domain or temporal holdouts.
    """
    if frame.hostname.nunique() < 20 or frame.label.value_counts().min() < 20 or frame.label.nunique() != 2:
        raise ValueError('Stratified 70/15/15 grouping requires 20 hosts and 20 examples per class')
    assignment = np.empty(len(frame), dtype=object)
    folds = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    for fold, (_, indices) in enumerate(folds.split(frame, frame.label, frame.hostname)):
        assignment[indices] = 'train' if fold < 14 else 'validation' if fold < 17 else 'test'
    return {name: frame.loc[assignment == name].reset_index(drop=True) for name in NAMES}


def verify(root):
    root = Path(root)
    processed = root / 'data/processed/urls'
    parts = {name: pd.read_parquet(root / 'data/splits/urls' / f'{name}.parquet') for name in NAMES}
    cleaned = pd.read_parquet(processed / 'urls_cleaned.parquet')
    external = pd.read_parquet(processed / 'phishtank_external.parquet')
    keys, hosts = set(), set()
    for name, frame in parts.items():
        assert not frame.empty, f'Empty {name} split'
        assert frame.label.isin([0, 1]).all() and frame.label.nunique() == 2
        assert frame.url.str.strip().ne('').all() and frame.url_original.str.strip().ne('').all()
        assert frame.url_key.is_unique
        assert not keys & set(frame.url_key), 'Cross-split URL leakage'
        assert not hosts & set(frame.hostname), 'Cross-split hostname leakage'
        assert frame.source.eq('phiusiil').all()
        assert frame.label.eq(1 - frame.original_label.astype(int)).all()
        keys.update(frame.url_key); hosts.update(frame.hostname)
    assert cleaned.url_key.is_unique and set(cleaned.url_key) == keys
    expected = cleaned.set_index('url_key').sort_index()
    actual = pd.concat(parts.values()).set_index('url_key').sort_index()
    pd.testing.assert_frame_equal(expected, actual)
    assert external.label.eq(1).all() and external.source.eq('phishtank').all()
    assert external.url_key.is_unique
    assert not keys & set(external.url_key) and not hosts & set(external.hostname)
    return dict(verified_records=len(cleaned), external_records=len(external),
                cross_split_url_overlap=0, cross_split_hostname_overlap=0,
                external_url_overlap=0, external_hostname_overlap=0,
                split_sizes={name: len(frame) for name, frame in parts.items()})


def completed(root):
    report = Path(root) / 'data/processed/urls/cleaning_report.json'
    if not report.exists():
        return False
    try:
        previous = json.loads(report.read_text())
        return previous.get('pipeline_version') == VERSION and previous.get('validation') == verify(root)
    except (OSError, ValueError, AssertionError, KeyError):
        return False


def run(root, seed=42):
    root = Path(root).resolve()
    if completed(root):
        logging.info('Existing outputs passed verification; skipping processing')
        return json.loads((root / 'data/processed/urls/cleaning_report.json').read_text())
    paths = discover(root)
    primary, primary_stats = ingest(paths['phiusiil'], 'phiusiil')
    external, external_stats = ingest(paths['phishtank'], 'phishtank')
    overlap = external.url_key.isin(primary.url_key)
    host_overlap = external.hostname.isin(primary.hostname)
    external_stats['removed']['overlap_primary_url'] = int(overlap.sum())
    external_stats['removed']['overlap_primary_hostname_additional'] = int((host_overlap & ~overlap).sum())
    external = external.loc[~(overlap | host_overlap)].reset_index(drop=True)
    external_stats['exported_retained'] = len(external)
    parts = split(primary, seed)
    processed = root / 'data/processed/urls'
    split_dir = root / 'data/splits/urls'
    processed.mkdir(parents=True, exist_ok=True); split_dir.mkdir(parents=True, exist_ok=True)
    # Only raw URL/label metadata is imported: supplied learned/content features
    # are deliberately excluded from the future feature-engineering inputs.
    primary.to_parquet(processed / 'urls_cleaned.parquet', index=False)
    external.to_parquet(processed / 'phishtank_external.parquet', index=False)
    for name, frame in parts.items():
        frame.to_parquet(split_dir / f'{name}.parquet', index=False)
    report = dict(pipeline_version=VERSION, seed=seed,
                  sources={'phiusiil':primary_stats, 'phishtank':external_stats},
                  label_mapping={'phiusiil':'original 1 -> legitimate 0; original 0 -> phishing 1', 'phishtank':'verified phishing -> 1'},
                  class_distribution={name: frame.label.value_counts().sort_index().to_dict() for name,frame in {'cleaned':primary, **parts}.items()},
                  split_method='20 stratified hostname-group folds, allocated 14/3/3; ratios approximate',
                  limitations=['Related subdomains can overlap; no registrable-domain or temporal guarantee.',
                               'PhishTank is phishing-only: measure phishing recall, not accuracy or false-positive rate.',
                               'Schemeless/non-HTTP(S) URLs rejected; malformed percent escapes retained as a lexical flag.',
                               'Original URL strings preserved; identity matching does not exhaust every server-specific URL equivalence.'],
                  validation=verify(root))
    report = json.loads(json.dumps(report))  # Stable JSON keys on initial and skipped runs.
    (processed / 'cleaning_report.json').write_text(json.dumps(report, indent=2)+'\n')
    logging.info('Completed successfully: %s', report['validation'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.verify:
        assert completed(args.root), 'Outputs incomplete, invalid, or missing completion report'
        print(json.dumps(json.loads((args.root/'data/processed/urls/cleaning_report.json').read_text())['validation'], indent=2))
    else:
        run(args.root, args.seed)

if __name__ == '__main__':
    main()
