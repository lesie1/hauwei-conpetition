"""All three roles use DeepSeek. Credentials and hidden reasoning are never logged."""
import json
import os
import socket
import http.client
import re
import time
import urllib.request
import urllib.error
from pathlib import Path
from common import ROOT,WORKFLOW,save,require


def parse_reply(text):
    s=text.strip();lines=s.splitlines()
    if len(lines)>=3 and lines[0].lower() in ('```json','```') and lines[-1]=='```':s='\n'.join(lines[1:-1])
    def pairs(items):
        obj={}
        for k,v in items:
            require(k not in obj,'Duplicate JSON key: '+k);obj[k]=v
        return obj
    def bad(value):raise ValueError('Nonfinite JSON number: '+value)
    out=json.loads(s,object_pairs_hook=pairs,parse_constant=bad)
    require(isinstance(out,dict),'Return one JSON object');return out


class ReplyTruncated(RuntimeError):
    """A generation limit is not a schema error and needs a different retry."""
    def __init__(self,max_tokens,usage):
        self.max_tokens=max_tokens
        self.usage=usage
        reasoning=usage.get('completion_tokens_details',{}).get('reasoning_tokens',0)
        super().__init__(f'DeepSeek reply reached max_tokens={max_tokens}; '
                         f'completion_tokens={usage.get("completion_tokens",0)}, reasoning_tokens={reasoning}')


class DeepSeek:
    def __init__(self,c):self.c=c
    def call(self,role,payload,folder,*,max_tokens=None):
        require(role in ('analysis','feature','code'),'Unknown agent role')
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        key=os.environ.get('DEEPSEEK_API_KEY');require(key,'DEEPSEEK_API_KEY missing; load your environment file')
        prompt=WORKFLOW.read_text(encoding='utf-8')+'\n'+(ROOT/'prompts'/f'{role}.txt').read_text(encoding='utf-8')
        prompt+='\nReturn exactly one valid JSON object following the requested template. Do not wrap the JSON in Markdown or add text outside it.'
        limit=self.c['agent_max_tokens'] if max_tokens is None else max_tokens
        require(type(limit) is int and limit>0,'Invalid generation token budget')
        body=dict(model=self.c[role+'_model'],messages=[dict(role='system',content=prompt),
            dict(role='user',content=json.dumps(payload,ensure_ascii=False,allow_nan=False))],
            response_format={'type':'json_object'},max_tokens=limit)
        if role=='feature' and payload.get('task')=='adaptation':
            body['reasoning_effort']=self.c.get('adaptation_reasoning_effort','low')
        save(folder/'request_without_credentials.json',body)
        req=urllib.request.Request(self.c['deepseek_base_url'].rstrip('/')+'/chat/completions',
            data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req,timeout=self.c['agent_timeout_seconds']) as res:answer=json.load(res)
                break
            except urllib.error.HTTPError as e:
                # Keep the service's reason, but never echo credentials or raw request bodies.
                details={}
                try:
                    parsed=json.loads(e.read(65536).decode('utf-8',errors='replace'))
                    error=parsed.get('error',{}) if isinstance(parsed,dict) else {}
                    if isinstance(error,dict):
                        for field in ('message','type','code','param'):
                            value=error.get(field)
                            if isinstance(value,(str,int,float)):
                                value=str(value).replace(key,'<REDACTED>')
                                value=re.sub(r'(?i)Bearer\s+\S+', 'Bearer <REDACTED>', value)
                                value=re.sub(r'\bsk-[A-Za-z0-9_-]+', '<REDACTED>', value)
                                details[field]=value[:1500]
                except (ValueError,OSError):pass
                save(folder/'network_error.json',dict(attempt=attempt+1,http_status=e.code,details=details))
                if e.code not in (408,429,500,502,503,504) or attempt==2:
                    message=details.get('message','No JSON error detail returned')
                    raise RuntimeError(f'DeepSeek HTTP {e.code}: {message}; session preserved') from None
            except (urllib.error.URLError,TimeoutError,socket.timeout,http.client.RemoteDisconnected,ConnectionResetError):
                save(folder/'network_error.json',dict(attempt=attempt+1,error='network unavailable'))
                if attempt==2:raise RuntimeError('DeepSeek network failed; restore network and resume this session') from None
            time.sleep(2**attempt*2)
        save(folder/'usage.json',answer.get('usage',{}));choice=answer['choices'][0]
        text=choice['message'].get('content') or ''
        (folder/'reply.txt').write_text(text,encoding='utf-8')
        finish=choice.get('finish_reason')
        save(folder/'response_meta.json',dict(model=answer.get('model'),finish_reason=finish,
            max_tokens=limit,reasoning_effort=body.get('reasoning_effort'),reply_characters=len(text)))
        if finish=='length':raise ReplyTruncated(limit,answer.get('usage',{}))
        if finish!='stop':raise RuntimeError(f'DeepSeek ended with finish_reason={finish}; session preserved')
        result=parse_reply(text);save(folder/'response.json',result);return result
