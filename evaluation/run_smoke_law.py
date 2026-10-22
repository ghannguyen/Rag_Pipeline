"""Collect three real API responses, then use the existing RAGAS scorer unchanged."""
import argparse,json,shutil,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from types import SimpleNamespace
import requests
from dotenv import load_dotenv
import evaluate_rag as ev

p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,default=Path('evaluation')/('smoke_law_'+datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).strftime('%Y%m%d_%H%M%S')))
p.add_argument('--judge-model',required=True)
p.add_argument('--collect-only',action='store_true')
p.add_argument('--score-saved',action='store_true')
a=p.parse_args();load_dotenv('.env');a.output.mkdir(parents=True,exist_ok=True)
args=SimpleNamespace(output=a.output,judge_model=a.judge_model,embedding_model='models/gemini-embedding-001',judge_timeout=240,top_k=15,top_n=5,threshold=.35,base_url='http://127.0.0.1:8001')
source=Path('evaluation/smoke_law_dataset.json')
records=json.loads(source.read_text());assert len(records)==3
if not a.score_saved:
 shutil.copy2(source,a.output/source.name)
 meta={'started':datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),'selected':3}
 rows=[]
 with requests.Session() as s:
  s.trust_env=False
  meta['schema_hash']=ev.check_openapi(s,args)
  for case in ev.load_cases(source,3):
   row={**case,'is_answerable':'Yes','answer':'','contexts':[],**dict.fromkeys(ev.METRICS),'status':'PENDING','error':'','context_source':'retrieval_search_pre_rerank','retrieval_seconds':0,'chat_seconds':0,'judge_seconds':0,'total_seconds':0}
   trace={'question':case['question']};t=time.monotonic()
   try:
    trace['retrieval']=ev.call_json(s,args.base_url+'/api/retrieval/search',{'query':case['question'],'top_k':15},360)
    row['contexts']=ev.parse_contexts(trace['retrieval']);row['retrieval_seconds']=time.monotonic()-t
    t2=time.monotonic()
    trace['chat']=ev.call_json(s,args.base_url+'/api/rag/chat',{'query':case['question'],'top_k':15,'top_n':5,'threshold':.35,'provider':'gemini'},360)
    row['answer']=ev.parse_answer(trace['chat']);row['chat_seconds']=time.monotonic()-t2
    row['status']='READY_TO_SCORE'
   except Exception as e: row['status']='API_ERROR';row['error']=ev.safe_error(e);trace['error']=row['error']
   row['total_seconds']=time.monotonic()-t
   (a.output/(case['test_case_id']+'_api.json')).write_text(json.dumps(trace,ensure_ascii=False,indent=2))
   rows.append(row);ev.save_outputs(rows,args,meta)
   print(case['test_case_id'],row['status'],len(row['contexts']),flush=True)
 (a.output/'rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 (a.output/'metadata.json').write_text(json.dumps(meta,indent=2))
else:
 rows=json.loads((a.output/'rows.json').read_text());meta=json.loads((a.output/'metadata.json').read_text())
if a.collect_only: raise SystemExit(0)
evaluator=ev.make_evaluator(args)
for row in rows:
 if row['status']!='READY_TO_SCORE':continue
 t=time.monotonic()
 try: ev.score_case(row,evaluator)
 except Exception as e: row['status']='JUDGE_ERROR';row['error']=ev.safe_error(e)
 row['judge_seconds']=time.monotonic()-t;row['total_seconds']+=row['judge_seconds']
 ev.save_outputs(rows,args,meta);(a.output/'rows.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 print(row['test_case_id'],row['status'],{m:row[m] for m in ev.METRICS},row['error'],flush=True)
 if row['error']:
  (a.output/'errors.json').write_text(json.dumps([{'id':r['test_case_id'],'error':r['error']} for r in rows if r['error']],ensure_ascii=False,indent=2))
  if any(c in row['error'] for c in ['503','429','UNAVAILABLE','RESOURCE_EXHAUSTED']):break
print('OUTPUT',a.output,flush=True)
