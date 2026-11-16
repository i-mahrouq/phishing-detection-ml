from pathlib import Path
import json
import pandas as pd
from src.preprocessing.emails import clean, urls, parse, keys, run


def test_indicators_and_html():
    raw='<html><script>danger()</script><a href="https://evil.test/login?a=1&amp;b=2">URGENT!!!</a></html>'
    assert urls(raw)==['https://evil.test/login?a=1&b=2']
    assert clean(raw)=='URGENT!!!'
    assert clean(' café \n verify!!! www.evil.test ')== 'café verify!!! www.evil.test'


def test_mime(tmp_path):
    p=tmp_path/'email'
    p.write_bytes(b'Subject: =?UTF-8?Q?caf=C3=A9?=\nFrom: a@test\nMIME-Version: 1.0\nContent-Type: multipart/alternative; boundary=x\n\n--x\nContent-Type: text/plain; charset=madeup\n\nhello\n--x\nContent-Type: text/html\n\n<a href="https://evil.test">hello</a>\n--x--\n')
    r=parse(p)
    assert r['subject']=='café' and 'hello' in r['body']
    assert urls(r['url_text'])==['https://evil.test']


def test_thread_keys():
    block=' '.join(f'word{i}' for i in range(100))
    assert keys(block) & keys('new reply\n\n'+block)
    assert keys('one','<id@test>') & keys('two',references='<id@test>')


def test_end_to_end(tmp_path):
    folder=tmp_path/'data/raw_data/emails';folder.mkdir(parents=True)
    rows=[{'label':i%2,'text':f'unique message token{i}'} for i in range(60)]
    rows += [rows[0],{'label':0,'text':'conflicting'},{'label':1,'text':'conflicting'},
             {'label':7,'text':'invalid'},{'label':0,'text':''}]
    pd.DataFrame(rows).to_csv(folder/'labeled.csv',index=False)
    a=run(tmp_path);b=run(tmp_path)
    assert a==b
    assert a['final_records']==60
    assert a['removed']=={'invalid_label':1,'empty_body':1,'conflicting_group':2,'exact_duplicate':1}
    assert a['validation']['verified_records']==60
    assert all(a['split_sizes'][s]>0 for s in ['train','validation','test'])


def test_large_fields_and_near_groups(tmp_path):
    folder=tmp_path/'data/raw_data/emails';folder.mkdir(parents=True)
    block=' '.join(f'phrase{i}' for i in range(100))
    rows=[{'label':i%2,'text':f'Independent email {i}'} for i in range(60)]
    rows += [{'label':0,'text':'first reply\n\n'+block},
             {'label':0,'text':'second reply\n\n'+block},
             {'label':1,'text':'longtext ' * 20000}]
    pd.DataFrame(rows).to_csv(folder/'labeled.csv',index=False)
    report=run(tmp_path)
    frame=pd.read_parquet(tmp_path/'data/processed/emails/emails_cleaned.parquet')
    variants=frame[frame.text.str.contains('reply')]
    assert len(variants)==2
    assert variants.duplicate_group.nunique()==1
    assert variants.split.nunique()==1
    assert report['final_records']==63


def test_infeasible_splits_replace_stale_outputs(tmp_path):
    folder=tmp_path/'data/raw_data/emails';folder.mkdir(parents=True)
    path=folder/'labeled.csv'
    pd.DataFrame([{'label':i%2,'text':f'content {i}'} for i in range(30)]).to_csv(path,index=False)
    run(tmp_path)
    pd.DataFrame([{'label':0,'text':'one group only'}]).to_csv(path,index=False)
    report=run(tmp_path)
    assert sum(report['split_sizes'].values())==1
    assert report['validation']['verified_records']==1


def test_url_schema_across_batches(tmp_path):
    folder=tmp_path/'data/raw_data/emails';folder.mkdir(parents=True)
    rows=[{'label':i%2,'text':f'Unique row {i} '+('https://example.test' if i<100 else 'no links')} for i in range(2100)]
    pd.DataFrame(rows).to_csv(folder/'labeled.csv',index=False)
    report=run(tmp_path)
    assert report['final_records']==2100
    assert report['validation']['verified_records']==2100
