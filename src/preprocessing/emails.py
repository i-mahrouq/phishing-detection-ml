"""Streaming ingestion, conservative duplicate grouping, and reproducible export.

Near duplicates are defined by matching substantial normalized paragraphs or quoted
blocks. This is a conservative heuristic, not exhaustive semantic similarity.
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import logging
from multiprocessing import Pool
from pathlib import Path
import re
import sqlite3
import unicodedata
from email import policy
from email.parser import BytesParser
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

URL = re.compile(r'(?:https?://|ftp://|www\.)[^\s<>"\'\x00-\x1f]+', re.I)

def digest(text):
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()

def clean(text):
    text = unicodedata.normalize('NFC', text or '')
    if re.search(r'</?(?:html|body|div|p|a|br|table)\b', text, re.I):
        soup = BeautifulSoup(text, 'lxml')
        for tag in soup(['script', 'style']):
            tag.decompose()
        text = soup.get_text(' ', strip=True)
    return re.sub(r'\s+', ' ', html.unescape(text)).strip()

def urls(text):
    # Preserve occurrences (including repeated links); HTML attributes are included.
    return [m.group().rstrip('.,;!?)') for m in URL.finditer(html.unescape(text))]

def keys(body, message_id='', references=''):
    """Grouping only: case/punctuation canonicalization never changes model text."""
    normalized = ' '.join(re.findall(r'\w+', body.casefold()))
    result = {'body:' + digest(normalized)} if normalized else set()
    # Long paragraphs and quoted/forwarded suffixes link thread variants.
    blocks = re.split(r'\n\s*\n|(?im:^-+\s*(?:original|forwarded) message.*$)', body)
    for block in blocks:
        words = re.findall(r'\w+', re.sub(r'(?m)^\s*>\s?', '', block).casefold())
        if len(words) >= 80:
            result.add('block:' + digest(' '.join(words)))
    for mid in re.findall(r'<[^<>\s]+>', message_id + ' ' + references):
        result.add('thread:' + mid)
    return result

def parse(path):
    with path.open('rb') as stream:
        msg = BytesParser(policy=policy.default).parse(stream)
    parts = [p for p in msg.walk() if p.get_content_maintype() == 'text'
             and p.get_content_subtype() in ('plain', 'html')
             and p.get_content_disposition() != 'attachment']
    # Prefer plain alternatives, while retaining URLs from every text part.
    plain, rich = [], []
    for part in parts:
        try:
            content = part.get_content(errors='replace')
        except (LookupError, UnicodeError):
            content = (part.get_payload(decode=True) or b'').decode('utf-8', 'replace')
        (plain if part.get_content_subtype() == 'plain' else rich).append(content)
    body = '\n\n'.join(plain or rich)
    headers = {k: str(v) for k, v in msg.items()}
    defects = [type(d).__name__ for p in msg.walk() for d in p.defects]
    return dict(subject=str(msg.get('subject', '')), sender=str(msg.get('from', '')),
                date=str(msg.get('date', '')), body=body, headers=json.dumps(headers),
                message_id=str(msg.get('message-id', '')),
                references=str(msg.get('references', ''))+' '+str(msg.get('in-reply-to', '')),
                url_text='\n'.join(plain + rich), defects=defects)

def discover(root):
    maildirs = sorted(p for p in root.rglob('maildir') if p.is_dir())
    files = sorted(p for p in root.rglob('*.csv') if '.ipynb_checkpoints' not in p.parts)
    compatible = []
    for p in files:
        with p.open(encoding='utf-8-sig', newline='') as f:
            header = next(csv.reader(f), [])
        if 'label' in header and ('text' in header or 'body' in header):
            compatible.append(p)
    if not maildirs:
        logging.warning('Missing Enron: expected data/raw_data/emails/enron/maildir')
    if not compatible:
        logging.warning('Missing labeled CSV: expected data/raw_data/emails/nazario/email_text.csv')
    if not maildirs and not compatible:
        raise FileNotFoundError('No email sources discovered')
    return maildirs, compatible

def prepare(item):
    item['_body'] = clean(item['body'])
    item['_text'] = clean(item.get('subject', '') + '\n' + item['body'])
    item['_keys'] = keys(item['body'], item.get('message_id', ''), item.get('references', ''))
    item['_urls'] = urls(item.get('subject', '') + '\n' + item.pop('url_text', item['body']))
    return item


def parse_worker(path):
    try:
        return str(path), prepare(parse(path))
    except Exception as exc:
        return str(path), RuntimeError(type(exc).__name__)


def run(root, seed=42, workers=4):
    csv.field_size_limit(2**31 - 1)
    root = Path(root).resolve()
    raw = root / 'data/raw_data/emails'
    maildirs, csvs = discover(raw)
    out = root / 'data/processed/emails'
    splits = root / 'data/splits/emails'
    out.mkdir(parents=True, exist_ok=True); splits.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(out / 'work.sqlite')
    db.execute('PRAGMA journal_mode=MEMORY')
    db.execute('PRAGMA synchronous=OFF')
    db.executescript('DROP TABLE IF EXISTS records; DROP TABLE IF EXISTS links; CREATE TABLE records (id INTEGER PRIMARY KEY, hash TEXT, label INTEGER, payload TEXT); CREATE TABLE links (key TEXT PRIMARY KEY, id INTEGER);')
    parent = []; exact = {}; counts = {}; removed = Counter()
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a != b: parent[max(a,b)] = min(a,b)
    def ingest(source, origin, label, item):
        c=counts.setdefault(source, Counter()); c['raw']+=1
        if isinstance(item, Exception):
            c['parsing_errors']+=1; removed['parsing_error']+=1; return
        c['parsed']+=1
        for field in ('subject','sender','date','body'):
            if not item.get(field, '').strip(): c['missing_'+field]+=1
        if item.get('defects'): c['malformed']+=1
        if str(label) not in ('0','1'):
            c['invalid_labels']+=1; removed['invalid_label']+=1; return
        item = item if '_body' in item else prepare(item)
        body=item['_body']
        if not body:
            removed['empty_body']+=1; return
        text=item['_text']
        idx=len(parent);parent.append(idx)
        # Body-only hashing catches copies with changed subjects.
        h=digest(body)
        if h in exact: union(idx,exact[h]); c['exact_duplicates']+=1
        else: exact[h]=idx
        for key in item['_keys']:
            previous=db.execute('SELECT id FROM links WHERE key=?',(key,)).fetchone()
            if previous: union(idx,previous[0])
            else: db.execute('INSERT INTO links VALUES (?,?)',(key,idx))
        links=item['_urls']
        payload=dict(record_id=digest(source+':'+origin),source=source,source_record=origin,
                     original_label=str(label),label=int(label),label_status='presumed_legitimate' if source.startswith('enron') else 'provided_binary_semantics_unverified',
                     subject=clean(item.get('subject','')),sender=item.get('sender',''),date=item.get('date',''),
                     headers=item.get('headers','{}'),text=text,body=body,urls=links,url_count=len(links),exact_hash=h)
        db.execute('INSERT INTO records VALUES (?,?,?,?)',(idx,h,int(label),json.dumps(payload)))
    for folder in maildirs:
        paths = sorted(p for p in folder.rglob('*') if p.is_file())
        with Pool(processes=workers) as pool:
            for path, item in tqdm(pool.imap(parse_worker, paths, chunksize=16), total=len(paths), desc='Enron'):
                ingest('enron', str(Path(path).relative_to(folder)), 0, item)
    for path in csvs:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for i,row in enumerate(tqdm(csv.DictReader(stream),desc=path.name),start=2):
                row['body']=row.get('text',row.get('body','')) or ''
                ingest(path.stem,str(i),row.get('label'),row)
    db.commit()
    metadata=list(db.execute('SELECT id,hash,label FROM records ORDER BY id'))
    groups={}; masks=Counter()
    for idx,h,label in metadata:
        g=find(idx);groups.setdefault(g,Counter())[label]+=1; masks[g]|=1 << label
    conflicts={g for g,m in masks.items() if m==3}
    kept=[];seen=set()
    for idx,h,label in metadata:
        if find(idx) in conflicts: removed['conflicting_group']+=1
        elif h in seen: removed['exact_duplicate']+=1
        else: kept.append((idx,find(idx),label));seen.add(h)
    if not kept: raise ValueError('No valid records remain')
    # Greedy class-stratified group allocation with fixed seed and size targets.
    sizes={}
    for _,g,label in kept: sizes.setdefault(g,np.zeros(2,dtype=int))[label]+=1
    totals=sum(sizes.values());targets=np.array([.70,.15,.15])[:,None]*totals
    current=np.zeros((3,2)); assignments={};rng=np.random.default_rng(seed)
    order=list(sizes);rng.shuffle(order);order.sort(key=lambda g:-sum(sizes[g]))
    for g in order:
        v=sizes[g]
        scores=[np.sum(((current + np.eye(3)[s,:,None]*v-targets)/np.maximum(targets,1))**2) for s in range(3)]
        s=int(np.argmin(scores));assignments[g]=s;current[s]+=v
    # Persist the expensive parsing/grouping stage before opening export files.
    db.executescript('DROP TABLE IF EXISTS export_groups; CREATE TABLE export_groups (id INTEGER PRIMARY KEY, group_id INTEGER, split INTEGER);')
    db.executemany('INSERT INTO export_groups VALUES (?,?,?)', [(idx,g,assignments[g]) for idx,g,label in kept])
    db.execute('CREATE TABLE IF NOT EXISTS run_counters (payload TEXT)')
    db.execute('DELETE FROM run_counters')
    db.execute('INSERT INTO run_counters VALUES (?)', (json.dumps(dict(sources=counts,removed=removed)),))
    db.commit()
    names=['train','validation','test'];writers={};batch=[];distribution=Counter();source_labels={}
    def flush():
        if not batch:return
        frame=pd.DataFrame(batch)
        schema = pa.Schema.from_pandas(frame, preserve_index=False)
        schema = schema.set(schema.get_field_index('urls'), pa.field('urls', pa.list_(pa.string())))
        for name,sub in [('cleaned',frame)]+[(n,frame[frame.split==n]) for n in names]:
            table=pa.Table.from_pandas(sub,schema=schema,preserve_index=False)
            url_index = table.schema.get_field_index('urls')
            table = table.set_column(url_index, 'urls', pa.array(sub.urls.tolist(), type=pa.list_(pa.string())))
            if name not in writers:
                path=out/'emails_cleaned.parquet' if name=='cleaned' else splits/(name+'.parquet')
                writers[name]=pq.ParquetWriter(path,table.schema,compression='zstd')
            writers[name].write_table(table)
        batch.clear()
    for idx,g,label in tqdm(kept,desc='Export'):
        record=json.loads(db.execute('SELECT payload FROM records WHERE id=?',(idx,)).fetchone()[0])
        record.update(duplicate_group=str(g),split=names[assignments[g]])
        batch.append(record);distribution[label]+=1
        source_labels.setdefault(record['source'],Counter())[label]+=1
        if len(batch)>=2000:flush()
    flush()
    for writer in writers.values():writer.close()
    near = len(kept) - len(sizes)
    for counters in counts.values():
        for metric in ('raw', 'parsed', 'parsing_errors', 'missing_subject', 'missing_sender', 'missing_date', 'missing_body', 'malformed', 'invalid_labels', 'exact_duplicates'):
            counters.setdefault(metric, 0)
    report=dict(seed=seed,workers=workers,sources=counts,removed=removed,valid_before_dedup=len(parent),final_records=len(kept),
                final_class_distribution=distribution,source_class_distribution=source_labels,
                exact_duplicate_records=sum(c['exact_duplicates'] for c in counts.values()),
                near_duplicate_extra_records_after_exact=near,conflicting_groups=len(conflicts),
                split_sizes=dict(zip(names,current.sum(axis=1).astype(int).tolist())),
                split_class_counts={n:current[i].astype(int).tolist() for i,n in enumerate(names)},
                limitations=['CSV numeric labels preserved; phishing versus spam semantics and exact file lineage unverified.',
                             'Enron is presumed legitimate; source and historical-era bias remain.',
                             'Paragraph/quoted-block/reference grouping is heuristic; undetected semantic or partial thread overlap can remain.',
                             'URL extraction covers explicit scheme/www links only; obfuscated and bare domains remain in text.',
                             'Use class weights and PR-AUC/per-class metrics; select any resampling using training only; evaluate source-specific performance.'])
    db.close()
    report['validation']=verify(out,splits)
    (out/'cleaning_report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def verify(out, splits):
    """Reopen exports in bounded batches and independently assert invariants."""
    groups = set()
    hashes = set()
    total = 0
    for name in ('train', 'validation', 'test'):
        partition_groups = set()
        partition_hashes = set()
        batches = pq.ParquetFile(splits / (name + '.parquet')).iter_batches(
            batch_size=4096, columns=['duplicate_group', 'exact_hash', 'label', 'text', 'body'])
        for batch in batches:
            frame = batch.to_pandas()
            gs = set(frame.duplicate_group)
            hs = set(frame.exact_hash)
            assert not groups & gs and not hashes & hs, 'Cross-split leakage'
            assert not partition_hashes & hs, 'Repeated exact body within split'
            assert not frame.exact_hash.duplicated().any()
            assert frame.label.isin([0, 1]).all()
            assert frame.text.str.strip().ne('').all() and frame.body.str.strip().ne('').all()
            partition_groups.update(gs)
            partition_hashes.update(hs)
            total += len(frame)
        groups.update(partition_groups)
        hashes.update(partition_hashes)
    assert pq.read_metadata(out / 'emails_cleaned.parquet').num_rows == total
    return dict(cross_split_duplicate_groups=0, cross_split_exact_hashes=0,
                invalid_labels=0, empty_text=0, verified_records=total)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();print(json.dumps(run(args.root,args.seed,args.workers),indent=2))

if __name__=='__main__':main()
