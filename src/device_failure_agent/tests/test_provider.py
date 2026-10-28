"""Offline regression tests; the HTTP boundary is an explicit test double."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(os.environ.get('DF_TEST_PROGRAM', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
import providers


class ProviderRegressionTests(unittest.TestCase):
    def config(self):
        return dict(deepseek_base_url='https://api.deepseek.com', agent_max_tokens=12000,
                    agent_timeout_seconds=1,
                    **{r+'_model': 'deepseek-v4-pro' for r in ('feature', 'analysis', 'code')})

    def success(self):
        return io.BytesIO(json.dumps({'choices': [{'message': {'content': '{"ok":true}'},
                            'finish_reason': 'stop'}], 'usage': {'total_tokens': 7}}).encode())

    def test_real_provider_messages_satisfy_json_contract_for_all_roles(self):
        def endpoint(req, timeout):
            body = json.loads(req.data)
            self.assertEqual(body['response_format'], {'type': 'json_object'})
            self.assertEqual(body['model'], 'deepseek-v4-pro')
            self.assertEqual(body['max_tokens'], 12000)
            messages = '\n'.join(m['content'] for m in body['messages']).lower()
            if 'json' not in messages:
                raise urllib.error.HTTPError(req.full_url, 400, 'Bad Request', {},
                    io.BytesIO(b'{"error":{"message":"Prompt must contain the word json to use response_format of type json_object","type":"invalid_request_error"}}'))
            return self.success()
        payload = dict(task='adaptation', domain='hdd', instructions='Hard disk failure prediction',
            schemas={'daily': {'columns': ['date', 'serial_number', 'failure', 'smart_5_raw']}},
            mapping_example=json.loads((ROOT/'examples/mapping_example.json').read_text(encoding='utf-8-sig')),
            complete_failure_coverage=False)
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only-secret'}), patch('urllib.request.urlopen', endpoint):
            for role in ('feature', 'analysis', 'code'):
                with self.subTest(role=role):
                    self.assertEqual(providers.DeepSeek(self.config()).call(role, payload, Path(temp)/role), {'ok': True})

    def test_400_reason_is_saved_and_shown_without_credentials_or_retry(self):
        secret = 'test-only-secret'
        error = {'error': {'message': 'Invalid format; '+secret+' Bearer hidden-value sk-hidden123',
                          'type': 'invalid_request_error', 'code': 'invalid_format', 'param': 'messages'}}
        def endpoint(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 400, 'Bad Request', {}, io.BytesIO(json.dumps(error).encode()))
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'DEEPSEEK_API_KEY': secret}), patch('urllib.request.urlopen', side_effect=endpoint) as request:
            with self.assertRaises(RuntimeError) as caught:
                providers.DeepSeek(self.config()).call('feature', {'task': 'adaptation'}, temp)
            saved = (Path(temp)/'network_error.json').read_text()
            self.assertEqual(request.call_count, 1)
            self.assertIn('Invalid format', str(caught.exception))
            self.assertIn('invalid_format', saved)
            self.assertEqual(json.loads(saved)['details']['param'], 'messages')
            for value in (secret, 'hidden-value', 'sk-hidden123'):
                self.assertNotIn(value, str(caught.exception)+saved)

    def test_non_json_error_body_is_not_exposed(self):
        def endpoint(req, timeout):
            raise urllib.error.HTTPError(req.full_url, 400, 'Bad Request', {}, io.BytesIO(b'<html>private-body</html>'))
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only-secret'}), patch('urllib.request.urlopen', endpoint):
            with self.assertRaises(RuntimeError) as caught:
                providers.DeepSeek(self.config()).call('feature', {'task': 'adaptation'}, temp)
            self.assertIn('No JSON error detail returned', str(caught.exception))
            self.assertNotIn('private-body', str(caught.exception)+(Path(temp)/'network_error.json').read_text())

    def test_retryable_error_can_recover_and_usage_is_preserved(self):
        failures = [urllib.error.HTTPError('https://api.deepseek.com/chat/completions', 429,
                     'Too Many Requests', {}, io.BytesIO(b'{"error":{"message":"rate limited"}}')),
                    self.success()]
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'test-only-secret'}), patch('urllib.request.urlopen', side_effect=failures) as request, patch('providers.time.sleep') as sleep:
            result = providers.DeepSeek(self.config()).call('feature', {'task': 'adaptation'}, temp)
            self.assertEqual(result, {'ok': True})
            self.assertEqual(request.call_count, 2)
            sleep.assert_called_once_with(2)
            self.assertEqual(json.loads((Path(temp)/'usage.json').read_text())['total_tokens'], 7)


if __name__ == '__main__':
    unittest.main()
