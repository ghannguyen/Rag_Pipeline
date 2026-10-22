"""Run exactly three valid Excel cases with the existing RAGAS scorer."""
import argparse, csv, json, time
from pathlib import Path
from types import SimpleNamespace
import requests
from dotenv import load_dotenv
import evaluate_rag as ev

p=argparse.ArgumentParser()
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();load_dotenv('.env')
config=json.loads((a.output/'run_config.json').read_text())
assert config['status']=='JUDGE_READY','Successful judge preflight required'
assert config['input_excel_path']==str(a.input.resolve())
assert not (a.output/'results_raw.csv').exists(),'Use a fresh output directory'
args=SimpleNamespace(output=a.output,judge_model=config['judge_model'],
    embedding_model=config['embedding_model'],judge_timeout=240,top_k=15,top_n=5,threshold=.35,
    base_url=config['backend_url'])
cases=ev.load_cases(a.input,3); assert len(cases)==3
meta={'started':config['timestamp'],'selected':3}
rows=[];errors=[]
def save(name,obj): (a.output/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def checkpoint():
    ev.save_outputs(rows,args,meta)
    save('rows.json',rows)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (a.output/'results_raw.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
    config['executed_testcases']=len(rows);save('run_config.json',config)
    if errors:save('errors.json',errors)

s=requests.Session();s.trust_env=False
meta['schema_hash']=ev.check_openapi(s,args)
evaluator=ev.make_evaluator(args)
config.update(retrieval_config={'top_k':15,'top_n':5,'threshold':.35},
    context_source='retrieval_search_pre_rerank',context_limitation=ev.CONTEXT_NOTE,
    metrics=['Faithfulness','ResponseRelevancy(strictness=1)','LLMContextPrecisionWithReference','LLMContextRecall'])
for case in cases:
    row={**case,'answer':'','contexts':[],**dict.fromkeys(ev.METRICS),'status':'PENDING','error':'',
        'context_source':'retrieval_search_pre_rerank','retrieval_seconds':0,'chat_seconds':0,'judge_seconds':0}
    trace={};start=time.monotonic();stage='retrieval'
    print('START',case['test_case_id'],flush=True)
    try:
        t=time.monotonic()
        trace_path=a.output/(case['test_case_id']+'_api.json')
        if trace_path.exists():
            trace=json.loads(trace_path.read_text())
            assert trace.get('chat',{}).get('query')==case['question']
        if 'retrieval' not in trace:
            trace['retrieval']=ev.call_json(s,args.base_url+'/api/retrieval/search',{'query':case['question'],'top_k':15},360)
        row['contexts']=ev.parse_contexts(trace['retrieval']);assert row['contexts'],'Empty contexts'
        row['retrieved_results']=trace['retrieval']['results']
        row['retrieved_sources']=[x.get('source') or x.get('metadata',{}).get('source') for x in row['retrieved_results']]
        row['retrieved_pages']=[x.get('metadata',{}).get('page',x.get('page')) for x in row['retrieved_results']]
        row['retrieval_seconds']=time.monotonic()-t
        stage='chat';t=time.monotonic()
        if 'chat' not in trace:
            trace['chat']=ev.call_json(s,args.base_url+'/api/rag/chat',{'query':case['question'],'top_k':15,'top_n':5,'threshold':.35,'provider':'gemini'},360)
        row['answer']=ev.parse_answer(trace['chat']);row['citations']=trace['chat'].get('citations',[])
        row['chat_seconds']=time.monotonic()-t
        save(case['test_case_id']+'_api.json',trace)
        stage='judge';t=time.monotonic();ev.score_case(row,evaluator);row['judge_seconds']=time.monotonic()-t
    except Exception as e:
        row['status']=stage.upper()+'_ERROR';row['error']=ev.safe_error(e)
    row['total_seconds']=time.monotonic()-start
    rows.append(row);save(case['test_case_id']+'_api.json',trace)
    if row['error']:errors.append({'id':case['test_case_id'],'stage':stage,'error':row['error']})
    checkpoint();print(row['test_case_id'],row['status'],{m:row[m] for m in ev.METRICS},row['error'],flush=True)
    if row['status']!='OK':
        config['status']='BLOCKED';break
else: config['status']='PASS' if all(r['status']=='OK' for r in rows) else 'METRIC_OR_API_ERROR'
config['pipeline_status_after']=s.get(args.base_url+'/api/pipeline/status',timeout=30).json()
checkpoint()
print('FINAL',config['status'],flush=True)
