import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from bridge import Bridge, incoming, mentions_bot, clean_reply, ai_payload, parse_model_reply


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'config.json'
        self.cfg = dict(app_id='test_app', self_wxid='test_self', mode='auto',
                        allowed_contacts=['test_friend'], ai_key='fake', ai_model='test',
                        ai_base='https://example.invalid/v1', wechat_token='fake',
                        wechat_base='https://example.invalid/wechat')
        self.path.write_text(json.dumps(self.cfg))
        self.bridge = Bridge(self.path, Path(self.tmp.name) / 'queue.db')
        self.event = dict(TypeName='AddMsg', Appid='test_app', Wxid='test_self', Data={
            'MsgType': 1, 'NewMsgId': 9876543210987654321, 'CreateTime': time.time(),
            'FromUserName': {'string': 'test_friend'}, 'ToUserName': {'string': 'test_self'},
            'Content': {'string': '你好'}})

    def status(self):
        with self.bridge.db() as db:
            return db.execute('SELECT status FROM messages').fetchone()[0]

    def test_generic_api_defaults_and_options(self):
        cfg = dict(ai_model='vendor-model', ai_provider='openai_compatible',
                   structured_replies=True, api_json_mode=False, ai_options={})
        messages = [{'role': 'user', 'content': '你好'}]
        payload = ai_payload(cfg, messages)
        self.assertNotIn('thinking', payload)
        self.assertNotIn('response_format', payload)
        self.assertNotIn('reasoning_split', payload)
        self.assertEqual(payload['model'], 'vendor-model')
        cfg.update(api_json_mode=True, ai_options={'max_tokens': 512})
        payload = ai_payload(cfg, messages)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(payload['max_tokens'], 512)
        cfg['ai_options'] = {'messages': []}
        with self.assertRaises(ValueError):
            ai_payload(cfg, messages)

    def test_duplicates_and_large_ids(self):
        self.assertEqual(self.bridge.accept(self.event), 'queued')
        self.assertEqual(self.bridge.accept(json.loads(json.dumps(self.event))), 'duplicate')
        self.assertTrue(incoming(self.event, self.cfg)[0].endswith('9876543210987654321'))

    def test_ignores_unapproved_self_group_and_old(self):
        for sender in ['stranger', 'test_self', 'group@chatroom']:
            event = copy.deepcopy(self.event)
            event['Data']['FromUserName']['string'] = sender
            self.assertEqual(self.bridge.accept(event), 'ignored')
        self.event['Data']['CreateTime'] -= 600
        self.assertEqual(self.bridge.accept(self.event), 'ignored')

    def test_paused_means_no_ai_and_no_send(self):
        self.cfg['mode'] = 'paused'
        self.path.write_text(json.dumps(self.cfg))
        with patch('bridge.request_json') as request:
            self.assertEqual(self.bridge.accept(self.event), 'paused')
            self.assertFalse(self.bridge.process_one())
            request.assert_not_called()

    def test_ai_then_send_exact_peer(self):
        self.bridge.accept(self.event)
        with patch('bridge.request_json', side_effect=[
            {'choices': [{'message': {'content': '你好，我是 AI 助理。'}}]}, {'ret': 200}
        ]) as request:
            self.bridge.process_one()
            self.assertEqual(request.call_args_list[1].args[1]['toWxid'], 'test_friend')
            self.assertEqual(self.status(), 'sent')
            self.assertFalse(self.bridge.process_one())
            self.assertEqual(request.call_count, 2)

    def test_uncertain_send_is_not_retried(self):
        self.bridge.accept(self.event)
        with patch('bridge.request_json', side_effect=[
            {'choices': [{'message': {'content': '你好'}}]}, TimeoutError()
        ]) as request:
            self.bridge.process_one()
            self.assertEqual(self.status(), 'needs_review')
            self.bridge.process_one()
            self.assertEqual(request.call_count, 2)

    def test_pause_during_ai_stops_send(self):
        self.bridge.accept(self.event)
        def fake_ai(*args):
            self.cfg['mode'] = 'paused'
            self.path.write_text(json.dumps(self.cfg))
            return {'choices': [{'message': {'content': '你好'}}]}
        with patch('bridge.request_json', side_effect=fake_ai) as request:
            self.bridge.process_one()
            self.assertEqual(request.call_count, 1)
            self.assertEqual(self.status(), 'skipped')

    def group_event(self):
        self.cfg.update(groups_enabled=True, allow_all_groups=True, bot_names=['黄白AI版'])
        self.path.write_text(json.dumps(self.cfg))
        event = copy.deepcopy(self.event)
        event['Data'].update(FromUserName={'string': 'test_room@chatroom'},
                             Content={'string': 'test_friend:\n@黄白AI版\u2005你好'},
                             MsgSource='<msgsource><atuserlist><![CDATA[,test_self]]></atuserlist></msgsource>')
        return event

    def test_group_mention_sends_to_group_without_history(self):
        event = self.group_event()
        item = incoming(event, self.cfg)
        self.assertEqual(item[1:3], ('test_room@chatroom', '你好'))
        self.assertEqual(self.bridge.accept(event), 'queued')
        with patch('bridge.request_json', side_effect=[
            {'choices': [{'message': {'content': '我是黄白的小助手。'}}]}, {'ret': 200}
        ]) as request:
            self.bridge.process_one()
            ai = request.call_args_list[0].args[1]
            self.assertEqual(len(ai['messages']), 2)
            self.assertTrue(ai['reasoning_split'])
            self.assertEqual(request.call_args_list[1].args[1]['toWxid'], 'test_room@chatroom')

    def test_group_normal_fake_at_all_and_self_ignored(self):
        event = self.group_event()
        for source in ['', '<msgsource/>', '<msgsource><atuserlist>notify@all</atuserlist></msgsource>',
                       '<msgsource><atuserlist>test_self_wrong</atuserlist></msgsource>']:
            event['Data']['MsgSource'] = source
            self.assertEqual(self.bridge.accept(event), 'ignored')
        event = self.group_event()
        event['Data']['Content']['string'] = 'test_self:\n@黄白AI版\u2005你好'
        self.assertEqual(self.bridge.accept(event), 'ignored')

    def test_kimi_group_reply_uses_compatible_parameters(self):
        self.cfg.update(ai_provider='moonshot', ai_model='kimi-k2.6',
                        ai_base='https://api.moonshot.cn/v1')
        self.bridge.accept(self.group_event())
        with patch('bridge.request_json', side_effect=[
            {'choices': [{'message': {'content': '我是黄白的小助手。'}}]}, {'ret': 200}
        ]) as request:
            self.bridge.process_one()
            url, body, headers = request.call_args_list[0].args
            self.assertEqual(url, 'https://api.moonshot.cn/v1/chat/completions')
            self.assertEqual(body['thinking'], {'type': 'disabled'})
            self.assertEqual(body['max_tokens'], 2048)
            self.assertNotIn('reasoning_split', body)
            self.assertEqual(request.call_args_list[1].args[1]['toWxid'], 'test_room@chatroom')
            self.assertEqual(self.status(), 'sent')

    def test_group_scope_can_be_revoked_during_ai(self):
        self.bridge.accept(self.group_event())
        def fake_ai(*args):
            self.cfg['groups_enabled'] = False
            self.path.write_text(json.dumps(self.cfg))
            return {'choices': [{'message': {'content': '你好'}}]}
        with patch('bridge.request_json', side_effect=fake_ai) as request:
            self.bridge.process_one()
            self.assertEqual(request.call_count, 1)
            self.assertEqual(self.status(), 'skipped')

    def test_observe_does_not_store_or_call_ai(self):
        event = self.group_event()
        self.cfg['mode'] = 'observe'
        self.path.write_text(json.dumps(self.cfg))
        with patch('bridge.request_json') as request:
            self.assertEqual(self.bridge.accept(event), 'observed')
            self.assertFalse(self.bridge.process_one())
            request.assert_not_called()
        with self.bridge.db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)

    def test_thinking_never_sent(self):
        self.assertEqual(clean_reply('<think>内部思考</think>你好'), '你好')
        with self.assertRaises(ValueError):
            clean_reply('<think>未结束')
        self.assertFalse(mentions_bot('<!DOCTYPE x><msgsource/>', 'test_self'))

    def test_memory_keeps_twenty_messages_and_separates_groups(self):
        self.cfg['memory_limit'] = 20
        for i in range(12):
            self.bridge.remember_sent({'id': str(i), 'peer': 'room_a@chatroom',
                                       'author': 'member_a', 'content': '问题'+str(i)},
                                      '回答'+str(i), self.cfg)
        with self.bridge.db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM memory').fetchone()[0], 20)
        history = self.bridge.history('room_a@chatroom', self.cfg)
        self.assertEqual(len(history), 18)
        self.assertIn('问题3', history[0]['content'])
        self.assertEqual(history[-1]['content'], '回答11')
        self.assertEqual(self.bridge.history('room_b@chatroom', self.cfg), [])

    def test_knowledge_context_keeps_memory_and_source_links(self):
        from knowledge import make_messages, attach_sources
        path = Path(self.tmp.name) / 'knowledge.json'
        path.write_text(json.dumps({'synced_at': '2026-09-15', 'chunks': [
            {'title': 'ChatCut 教程', 'url': 'https://my.feishu.cn/docx/test',
             'text': '在 Codex 中安装 ChatCut 插件。'}]}))
        cfg = dict(self.cfg, knowledge_enabled=True, knowledge_path=str(path))
        history = [{'role': 'user', 'content': 'ChatCut是什么'},
                   {'role': 'assistant', 'content': '剪辑工具'}]
        messages, sources = make_messages(cfg, '那个怎么安装', '你是黄白的小助手。', history)
        self.assertEqual(messages[1:3], history)
        self.assertIn('安装 ChatCut', messages[-1]['content'])
        self.assertIn('https://my.feishu.cn/docx/test', attach_sources('可以看这篇[发链接1]', sources))
        self.assertNotIn('https://', attach_sources('装个插件就行[资料1]', sources))
        self.assertEqual(attach_sources('这个我不确定', sources), '这个我不确定')

    def test_casual_chat_does_not_search_knowledge(self):
        from knowledge import make_messages
        with patch('knowledge.search') as search:
            messages, sources = make_messages(dict(self.cfg, knowledge_enabled=True,
                knowledge_path='/not/used'), '你觉得今天怎么样', '说人话')
            search.assert_not_called()
            self.assertEqual(messages[-1]['content'], '你觉得今天怎么样')
            self.assertEqual(sources, [])

    def test_silence_never_sends_a_message(self):
        self.bridge.accept(self.event)
        with patch('bridge.request_json', return_value={
            'choices': [{'message': {'content': '[[不回复]]'}}]}) as request:
            self.bridge.process_one()
            self.assertEqual(request.call_count, 1)
            self.assertEqual(self.status(), 'silent')

    def test_mention_gets_response_even_when_model_is_silent(self):
        self.cfg['allow_model_silence'] = False
        event = self.group_event()
        event['Data']['Content']['string'] = 'test_friend:\n@黄白AI版\u2005为什么不回复我'
        self.bridge.accept(event)
        with patch('bridge.request_json', side_effect=[
            {'choices': [{'message': {'content': '[[不回复]]'}}]}, {'ret': 200}
        ]) as request:
            self.bridge.process_one()
            self.assertEqual(self.status(), 'sent')
            self.assertEqual(request.call_count, 2)
            self.assertEqual(request.call_args_list[1].args[1]['toWxid'], 'test_room@chatroom')
            self.assertEqual(request.call_args_list[1].args[1]['content'], '在，这条我暂时不知道怎么接。')
            self.assertEqual(self.bridge.observations['silence_fallbacks'], 1)

    def test_explicit_quiet_request_does_not_call_model_or_send(self):
        self.cfg['allow_model_silence'] = False
        event = self.group_event()
        event['Data']['Content']['string'] = 'test_friend:\n@黄白AI版\u2005请别回复这条消息。'
        self.bridge.accept(event)
        with patch('bridge.request_json') as request:
            self.bridge.process_one()
            request.assert_not_called()
            self.assertEqual(self.status(), 'silent')

    def test_structured_output_only_extracts_final_reply(self):
        cfg = dict(self.cfg, structured_replies=True)
        messages = [{'role': 'system', 'content': '有态度'}, {'role': 'assistant', 'content': '你好'}]
        payload = ai_payload(cfg, messages)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertEqual(json.loads(payload['messages'][1]['content']), {'reply': '你好'})
        self.assertEqual(messages[0]['content'], '有态度')
        result = {'choices': [{'message': {'content': '{"reply":"我叫小白。"}'}, 'finish_reason': 'stop'}]}
        self.assertEqual(parse_model_reply(result, cfg), '我叫小白。')
        for raw in ['分析：\n需要回应，有态度', '{"analysis":"思考","reply":"你好"}',
                    '{"reply":"需要回应，保持性格。最终决定：你好"}']:
            result['choices'][0]['message']['content'] = raw
            with self.assertRaises(ValueError):
                parse_model_reply(result, cfg)
        result['choices'][0].update(message={'content': '{"reply":"你好"}'}, finish_reason='length')
        with self.assertRaises(ValueError):
            parse_model_reply(result, cfg)

    def test_leaked_draft_is_regenerated_before_single_send(self):
        self.cfg['structured_replies'] = True
        self.path.write_text(json.dumps(self.cfg))
        self.bridge.accept(self.event)
        with patch('bridge.request_json', side_effect=[
            {'choices': [{'message': {'content': '{"reply":"需要回应，保持性格。回应方向：先判断"}'}}]},
            {'choices': [{'message': {'content': '{"reply":"在，叫我小白就行。"}'}}]},
            {'ret': 200}
        ]) as request:
            self.bridge.process_one()
            self.assertEqual(self.status(), 'sent')
            self.assertEqual(request.call_count, 3)
            self.assertEqual(request.call_args_list[2].args[1]['content'], '在，叫我小白就行。')
            self.assertNotIn('先判断', str(request.call_args_list[1].args[1]))

    def test_failed_repair_still_sends_no_draft(self):
        self.bridge.accept(self.event)
        with patch('bridge.request_json', side_effect=[
            {'choices': [{'message': {'content': '需要回应，保持性格。最终决定：'}}]},
            TimeoutError(), {'ret': 200}
        ]) as request:
            self.bridge.process_one()
            self.assertEqual(self.status(), 'sent')
            self.assertEqual(request.call_args_list[2].args[1]['content'], '这条我暂时没答好，先不瞎说。')

    def test_draft_history_is_excluded_without_losing_normal_analysis(self):
        self.cfg['memory_limit'] = 20
        self.bridge.remember_sent({'id': 'a', 'peer': 'room', 'author': '', 'content': '你是谁'},
                                  '需要回应，保持性格。可能的回应：小白', self.cfg)
        self.bridge.remember_sent({'id': 'b', 'peer': 'room', 'author': '', 'content': '分析下这段代码'},
                                  '分析：这里循环结束后还要返回结果。', self.cfg)
        self.assertEqual(self.bridge.history('room', self.cfg), [
            {'role': 'user', 'content': '分析下这段代码'},
            {'role': 'assistant', 'content': '分析：这里循环结束后还要返回结果。'}])


if __name__ == '__main__':
    unittest.main()
