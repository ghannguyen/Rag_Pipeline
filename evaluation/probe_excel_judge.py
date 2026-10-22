"""Read-only Excel/backend/Gemini preflight. Does not call any testcase."""
import json, os, time, argparse, importlib.metadata
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from dotenv import load_dotenv
import evaluate_rag as ev

p=argparse.ArgumentParser()
p.add_argument('--input', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--judge-model', default=None)
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); load_dotenv('.env')
def save(name,data):
    (a.output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
cases=ev.load_cases(a.input,3)
save('selected_excel_rows.json',cases)
config=dict(input_excel_path=str(a.input.resolve()),sheets=list(dict.fromkeys(c['sheet'] for c in cases)),
    number_testcases=3,executed_testcases=0,backend_url='http://127.0.0.1:8001',judge_model=None,
    embedding_model='models/gemini-embedding-001',embedding_task='SEMANTIC_SIMILARITY',
    judge_config={'temperature':0,'max_output_tokens':8192},
    ragas_version=importlib.metadata.version('ragas'),timestamp=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).isoformat(),
    status='PREFLIGHT',pdf_files=[p.name for p in sorted(Path('data/raw').glob('*.pdf'))])
save('run_config.json',config)
s=requests.Session();s.trust_env=False
config['health']=s.get(config['backend_url']+'/health',timeout=20).json()
config['pipeline_status']=s.get(config['backend_url']+'/api/pipeline/status',timeout=30).json()
assert config['health']['status']=='ok'
assert config['pipeline_status']['chunks']==config['pipeline_status']['vectors']==695
key=os.getenv('GOOGLE_API_KEY') or os.getenv('GEMINI_API_KEY')
assert key,'API key missing'
headers={'x-goog-api-key':key}; base='https://generativelanguage.googleapis.com/v1beta/'
errors=[]
def request(method,path,**kwargs):
    for attempt in range(4):
        r=s.request(method,base+path,headers=headers,timeout=60,**kwargs)
        if r.ok: return r.json()
        error={'path':path,'attempt':attempt+1,'http_status':r.status_code,'body':r.text.replace(key,'[REDACTED]')[:2000]}
        errors.append(error);save('errors.json',errors);print(json.dumps(error,ensure_ascii=False),flush=True)
        if r.status_code not in (429,503) or attempt==3: return None
        time.sleep(2**(attempt+1))
models=request('GET','models')
save('models.json',models)
if models:
    available=[m['name'] for m in models.get('models',[]) if 'generateContent' in m.get('supportedGenerationMethods',[])]
    preferred=os.getenv('LLM_MODEL','')
    candidates=sorted(available,key=lambda n:(n!='models/'+preferred, 'flash' not in n, 'preview' in n,n))
    if a.judge_model:
        assert a.judge_model in available,'Requested model not in live models/list'
        candidates=[a.judge_model]
    print('AVAILABLE',candidates,flush=True)
    if candidates:
        model=candidates[0];config['judge_model']=model
        result=request('POST',model+':generateContent',json={'contents':[{'parts':[{'text':'Return JSON with ok equal to true.'}]}],
            'generationConfig':{'temperature':0,'maxOutputTokens':8192,'responseMimeType':'application/json',
                'responseSchema':{'type':'OBJECT','properties':{'ok':{'type':'BOOLEAN'}},'required':['ok']}}})
        save('judge_probe.json',result)
        try:
            candidate=result['candidates'][0]
            reply=''.join(part.get('text','') for part in candidate['content']['parts'])
            assert candidate['finishReason']=='STOP' and json.loads(reply)=={'ok':True}
            config['status']='JUDGE_READY'
        except (TypeError,KeyError,ValueError,AssertionError) as exc:
            config['status']='BLOCKED'
            errors.append({'stage':'judge_probe','error':'Probe did not return complete valid JSON'})
            save('errors.json',errors)
    else: config['status']='BLOCKED'
else: config['status']='BLOCKED'
save('run_config.json',config)
print('STATUS',config['status'],flush=True)
raise SystemExit(0 if config['status']=='JUDGE_READY' else 2)
