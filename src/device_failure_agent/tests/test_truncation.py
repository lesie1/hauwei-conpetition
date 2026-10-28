"""Replay the captured length-stop pattern through the real provider and controller."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(os.environ['DF_TEST_PROGRAM']).resolve() if 'DF_TEST_PROGRAM' in os.environ else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agent import Loop
from providers import DeepSeek


class TruncationRecoveryTests(unittest.TestCase):
    def setup_loop(self, root, **overrides):
        c=dict(deepseek_base_url='https://api.deepseek.com',agent_max_tokens=12000,
               agent_max_tokens_cap=65536,max_truncation_retries=1,adaptation_reasoning_effort='low',
               agent_timeout_seconds=1,max_agent_calls=20,max_reported_tokens=1500000,max_repairs=2,
               **{r+'_model':'deepseek-v4-pro' for r in ('feature','analysis','code')})
        c.update(overrides)
        loop=Loop.__new__(Loop);loop.root=Path(root);loop.c=c
        loop.s={'calls':4,'tokens':50925,'trials':[],'pending':None}
        loop.provider=DeepSeek(c)
        return loop

    def response(self, reason='stop', content='{"ok":true}', completion=6, reasoning=0):
        return io.BytesIO(json.dumps({'model':'deepseek-v4-pro','choices':[{'finish_reason':reason,
            'message':{'content':content,'reasoning_content':'PRIVATE_REASONING_DO_NOT_PERSIST'}}],
            'usage':{'prompt_tokens':4943,'completion_tokens':completion,'total_tokens':4943+completion,
                     'completion_tokens_details':{'reasoning_tokens':reasoning}}}).encode())

    def test_length_changes_budget_and_preserves_every_call_and_token(self):
        requests=[]
        def endpoint(req,timeout):
            b=json.loads(req.data);requests.append(b)
            if b['max_tokens']<24000:
                return self.response('length','',12000,12000)
            return self.response()
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ,{'DEEPSEEK_API_KEY':'offline-test-secret'}),patch('urllib.request.urlopen',endpoint):
            loop=self.setup_loop(temp)
            answer=loop.validated_call('feature',{'task':'adaptation'},lambda a:a)
            self.assertEqual(answer,{'ok':True})
            self.assertEqual([r['max_tokens'] for r in requests],[12000,24000])
            self.assertTrue(all('repair' not in json.loads(r['messages'][1]['content']) for r in requests))
            self.assertEqual(loop.s['calls'],6)
            self.assertEqual(loop.s['tokens'],50925+16943+4949)
            self.assertEqual(json.loads((Path(temp)/'calls/00005_feature/response_meta.json').read_text())['finish_reason'],'length')
            events=(Path(temp)/'events.jsonl').read_text()
            self.assertIn('reply_budget_increased',events)
            self.assertNotIn('response_repair',events)
            for file in Path(temp).rglob('*'):
                if file.is_file(): self.assertNotIn('PRIVATE_REASONING_DO_NOT_PERSIST',file.read_text(encoding='utf-8'))

    def test_budget_cap_stops_without_repeating_identical_request(self):
        calls=[]
        def endpoint(req,timeout):
            calls.append(json.loads(req.data));return self.response('length','',12000,12000)
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'DEEPSEEK_API_KEY':'offline-test-secret'}),patch('urllib.request.urlopen',endpoint):
            loop=self.setup_loop(temp,agent_max_tokens_cap=12000)
            with self.assertRaises(RuntimeError) as caught:
                loop.validated_call('feature',{'task':'adaptation'},lambda a:a)
            self.assertIn('12000',str(caught.exception))
            self.assertEqual(len(calls),1)
            self.assertEqual(loop.s['tokens'],50925+16943)

    def test_adaptation_effort_is_explicit_but_other_roles_keep_their_default(self):
        requests=[]
        def endpoint(req,timeout):
            requests.append(json.loads(req.data));return self.response()
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'DEEPSEEK_API_KEY':'offline-test-secret'}),patch('urllib.request.urlopen',endpoint):
            loop=self.setup_loop(temp)
            loop.call('feature',{'task':'adaptation'})
            loop.call('feature',{'task':'features'})
            loop.call('analysis',{'task':'analysis'})
            self.assertEqual(requests[0]['reasoning_effort'],'low')
            self.assertNotIn('reasoning_effort',requests[1])
            self.assertNotIn('reasoning_effort',requests[2])

    def test_length_retry_respects_global_call_and_token_limits(self):
        for limit in ('calls','tokens'):
            with self.subTest(limit=limit),tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'DEEPSEEK_API_KEY':'offline-test-secret'}),patch('urllib.request.urlopen') as endpoint:
                endpoint.side_effect=lambda req,timeout:self.response('length','',12000,12000)
                c={'max_agent_calls':5} if limit=='calls' else {'max_reported_tokens':50926}
                loop=self.setup_loop(temp,**c)
                with self.assertRaises(RuntimeError):
                    loop.validated_call('feature',{'task':'adaptation'},lambda a:a)
                self.assertEqual(endpoint.call_count,1)
                self.assertEqual(loop.s['calls'],5)

    def test_non_length_stop_is_not_mislabeled_or_repaired(self):
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'DEEPSEEK_API_KEY':'offline-test-secret'}),patch('urllib.request.urlopen',side_effect=lambda req,timeout:self.response('content_filter','')) as endpoint:
            loop=self.setup_loop(temp)
            with self.assertRaises(RuntimeError) as caught:
                loop.validated_call('feature',{'task':'adaptation'},lambda a:a)
            self.assertIn('content_filter',str(caught.exception))
            self.assertEqual(endpoint.call_count,1)


if __name__=='__main__':unittest.main()
