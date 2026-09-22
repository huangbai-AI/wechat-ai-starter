import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from start import App, make_config, model_fingerprint, handler


def inputs(scene='private'):
    return dict(app_id='demo-app', self_wxid='demo-bot', ai_model='demo-model', ai_base='https://model.example.com/v1',
                ai_key='DEMO_MODEL_SECRET', wechat_token='DEMO_WECHAT_SECRET',
                wechat_base='https://wechat.example.com/api', scene=scene, bot_name='小白',
                allowed_contacts='demo-friend', allowed_groups='demo-room@chatroom', ai_options='{}')


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.app = App(self.tmp.name, 0)
        self.addCleanup(self.app.close)

    def test_private_group_both_and_no_self(self):
        for scene in ('private', 'group', 'both'):
            cfg = make_config(inputs(scene), {}, 8819)
            self.assertEqual(bool(cfg['allowed_contacts']), scene != 'group')
            self.assertEqual(cfg['groups_enabled'], scene != 'private')
            self.assertFalse(cfg['allow_all_groups'])
        data=inputs();data['allowed_contacts']='demo-bot'
        with self.assertRaises(ValueError): make_config(data, {}, 8819)

    def test_credentials_preserved_but_not_returned(self):
        self.app.action('save', inputs())
        public = json.dumps(self.app.public_state())
        self.assertNotIn('DEMO_MODEL_SECRET', public)
        self.assertNotIn('DEMO_WECHAT_SECRET', public)
        self.assertNotIn(self.app.config()['webhook_secret'], public)
        data=inputs();data.update(ai_key='',wechat_token='')
        self.app.action('save', data)
        self.assertEqual(self.app.config()['ai_key'],'DEMO_MODEL_SECRET')

    def test_invalid_config_keeps_previous_config(self):
        self.app.action('save', inputs())
        before=self.app.config();bad=inputs();bad['ai_base']='https://model.example.com/v1/chat/completions'
        with self.assertRaises(ValueError):self.app.action('save',bad)
        self.assertEqual(before,self.app.config())

    def test_enable_requires_model_connection_and_incoming(self):
        self.app.action('save', inputs())
        with self.assertRaisesRegex(ValueError,'检查模型'):self.app.action('enable',{})
        self.app.checked_model=model_fingerprint(self.app.config())
        with self.assertRaisesRegex(ValueError,'连接微信'):self.app.action('enable',{})
        cfg=self.app.config();cfg['callback_configured']=True;self.app.save(cfg)
        self.app.tunnel=Mock();self.app.tunnel.poll.return_value=None
        with self.assertRaisesRegex(ValueError,'识别到'):self.app.action('enable',{})
        self.app.bridge.observations['eligible']=1
        self.app.action('enable',{});self.assertEqual(self.app.config()['mode'],'auto')
        self.app.action('pause',{});self.assertEqual(self.app.config()['mode'],'paused')
        self.app.tunnel=None

    def test_callback_requires_explicit_checkbox(self):
        self.app.action('save',inputs())
        with patch.object(self.app,'connect') as connect:
            with self.assertRaises(ValueError):self.app.action('connect',{})
            connect.assert_not_called()

    def test_lookup_before_save_does_not_enable_or_store_contacts(self):
        with patch('start.request_json', return_value={'ret':200,'data':{'wxid':'demo-bot','nickName':'小白','mobile':'not-returned'}}):
            result=self.app.action('lookup_self', inputs())
            self.assertEqual(result['self_wxid'],'demo-bot')
            self.assertNotIn('mobile',result)
        responses=[{'ret':200,'data':{'friends':['demo-friend'],'chatrooms':[]}},
                   {'ret':200,'data':[{'userName':'demo-friend','nickName':'测试好友','remark':'小明'}]}]
        with patch('start.request_json',side_effect=responses):
            result=self.app.action('lookup_contacts',dict(inputs(),lookup_kind='private'))
        self.assertEqual(result['contacts'],[{'id':'demo-friend','name':'小明'}])
        self.assertFalse(self.app.path.exists())

    def test_restart_returns_to_observe(self):
        cfg=make_config(inputs(),{},0);cfg.update(mode='auto',callback_configured=True,public_base='https://old.example.com')
        self.app.save(cfg)
        restarted=App(self.tmp.name,0)
        self.assertEqual(restarted.config()['mode'],'observe')
        self.assertFalse(restarted.config()['callback_configured'])

    def test_local_post_security_and_webhook_to_private_reply(self):
        srv=ThreadingHTTPServer(('127.0.0.1',0),handler(self.app,0,'demo-csrf'))
        port=srv.server_address[1];srv.RequestHandlerClass=handler(self.app,port,'demo-csrf')
        thread=threading.Thread(target=srv.serve_forever,daemon=True);thread.start()
        self.addCleanup(srv.server_close);self.addCleanup(srv.shutdown)
        base=f'http://127.0.0.1:{port}'
        raw=json.dumps(dict(inputs(),action='save')).encode()
        bad=Request(base+'/action',data=raw,headers={'Content-Type':'application/json'})
        with self.assertRaises(HTTPError) as error:urlopen(bad)
        self.assertEqual(error.exception.code,403)
        error.exception.close()
        req=Request(base+'/action',data=raw,headers={'Content-Type':'application/json','Origin':base,'X-CSRF-Token':'demo-csrf'})
        self.assertTrue(json.load(urlopen(req))['ok'])
        cfg=self.app.config()
        event=dict(TypeName='AddMsg',Appid='demo-app',Wxid='demo-bot',Data=dict(MsgType=1,NewMsgId=1234567890123456789,CreateTime=time.time(),FromUserName={'string':'demo-friend'},ToUserName={'string':'demo-bot'},Content={'string':'你好'}))
        url=f'http://127.0.0.1:{self.app.server.server_address[1]}/wechat/'+cfg['webhook_secret']
        event_request=lambda:Request(url,data=json.dumps(event).encode(),headers={'Content-Type':'application/json'})
        self.assertEqual(json.load(urlopen(event_request()))['state'],'observed')
        self.assertEqual(self.app.public_state()['eligible'],1)
        cfg['mode']='auto';self.app.save(cfg)
        with patch('bridge.request_json',side_effect=[{'choices':[{'finish_reason':'stop','message':{'content':'{"reply":"你好，我已收到。"}'}}]}, {'ret':200}]) as external:
            self.assertEqual(json.load(urlopen(event_request()))['state'],'queued')
            deadline=time.monotonic()+3
            while self.app.public_state()['sent']!=1 and time.monotonic()<deadline:time.sleep(.05)
            self.assertEqual(self.app.public_state()['sent'],1)
            self.assertEqual(external.call_args_list[-1].args[1]['toWxid'],'demo-friend')
        self.assertEqual(json.load(urlopen(event_request()))['state'],'duplicate')
        self.assertEqual(self.app.public_state()['sent'],1)

if __name__=='__main__':unittest.main()
