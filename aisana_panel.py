"""Phone control panel for Aisana:  python3 aisana_panel.py   ->  http://<jetson>:8080

Buttons: start / stop Aisana, emergency STOP, wave, dance, "say this", "ask her",
live status (arms, legs, camera, internet) and the last log lines.
Python 3.6 standard library only (runs with the system python3 on the Jetson).
No login (the owner's choice): reach it over the robot's Wi-Fi, Tailscale or a
tunnel, and keep the link to yourselves.
"""
import json
import os
import re
import socketserver
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

import aisana_control

HERE = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get('AISANA_PANEL_PORT', '8080'))
LOG = os.path.join(HERE, 'realtime.log')
PATTERN = 'python[0-9.]* -u aisana_realtime[.]py'
NOISE = re.compile(r'GST_ARGUS|CONSUMER|Exposure Range|Camera (?:index|mode)|Output Stream|seconds to Run|'
                   r'Frame Rate|WARN:0|^\s*$')


def running():
    return subprocess.call(['pgrep', '-f', PATTERN], stdout=subprocess.DEVNULL) == 0


def log_tail(lines=40):
    try:
        with open(LOG, 'rb') as handle:
            handle.seek(0, 2)
            handle.seek(max(0, handle.tell() - 60000))
            text = handle.read().decode('utf-8', 'replace')
    except OSError:
        return []
    return [line for line in text.splitlines() if not NOISE.search(line)][-lines:]


def internet():
    try:
        route = subprocess.check_output(['ip', 'route', 'get', '8.8.8.8'], stderr=subprocess.DEVNULL)
        device = re.search(r'\bdev (\S+)', route.decode())
        name = device.group(1) if device else '?'
    except (OSError, subprocess.CalledProcessError):
        return 'нет'
    return {'eth1': 'роутер Beeline (USB)', 'wlan0': 'Wi-Fi'}.get(name, name)


def status():
    state = {'running': running(), 'internet': internet(), 'log': log_tail()}
    if state['running']:
        state['aisana'] = aisana_control.send({'action': 'status'}) or {'ok': False, 'error': 'starting'}
    return state


def start():
    if running():
        return {'ok': True, 'note': 'already running'}
    with open(os.devnull, 'wb') as null:
        subprocess.Popen(['setsid', 'nohup', './start.sh'], cwd=HERE, stdout=null, stderr=null,
                         stdin=subprocess.DEVNULL)
    return {'ok': True}


def stop():
    subprocess.call(['./stop.sh'], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {'ok': True}


PAGE = r"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Айсана — пульт</title>
<style>
 body{font-family:-apple-system,system-ui,sans-serif;margin:0;background:#0f1115;color:#e8e8ea}
 header{padding:16px 18px;font-size:20px;font-weight:600;display:flex;justify-content:space-between;align-items:center}
 .dot{width:12px;height:12px;border-radius:50%;display:inline-block;margin-right:8px;background:#666}
 .on{background:#3ecf6e}.off{background:#e5484d}
 main{padding:0 14px 30px;max-width:640px;margin:auto}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0}
 button{border:0;border-radius:14px;padding:16px;font-size:17px;font-weight:600;color:#fff;background:#2b2f3a}
 button:active{transform:scale(.98)}
 .stop{background:#e5484d;font-size:22px;grid-column:span 2;padding:22px}
 .go{background:#3b82f6}.off-btn{background:#444a57}
 .card{background:#181b22;border-radius:14px;padding:12px 14px;margin:10px 0}
 .row{display:flex;justify-content:space-between;padding:4px 0;font-size:15px}
 .row span:last-child{color:#b8bcc8;text-align:right;max-width:60%}
 textarea{width:100%;box-sizing:border-box;border-radius:12px;border:0;padding:12px;font-size:16px;background:#232733;color:#fff;min-height:64px}
 pre{white-space:pre-wrap;font-size:12px;line-height:1.35;color:#aeb3c2;max-height:340px;overflow:auto;margin:0}
 #msg{min-height:20px;font-size:14px;color:#9aa0ad;margin:6px 2px}
</style></head><body>
<header><span><span id="dot" class="dot"></span>Айсана</span><span id="net" style="font-size:13px;color:#9aa0ad"></span></header>
<main>
 <div class="grid">
  <button class="stop" onclick="act('stop')">СТОП</button>
  <button class="go" onclick="post('/api/start')">Включить</button>
  <button class="off-btn" onclick="if(confirm('Выключить Айсану?'))post('/api/stop')">Выключить</button>
  <button onclick="act('wave')">👋 Помахать</button>
  <button onclick="act('dance')">💃 Танцевать</button>
 </div>
 <div class="card">
  <textarea id="text" placeholder="Текст для Айсаны"></textarea>
  <div class="grid"><button onclick="talk('say')">Сказать это</button><button onclick="talk('ask')">Спросить её</button></div>
 </div>
 <div id="msg"></div>
 <div class="card" id="state"></div>
 <div class="card"><pre id="log"></pre></div>
</main>
<script>
const msg=t=>{document.getElementById('msg').textContent=t};
async function post(url,body){try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});const j=await r.json();msg(j.ok?'Готово':'Ошибка: '+(j.error||''));setTimeout(refresh,800)}catch(e){msg('Нет связи с роботом')}}
function act(a){post('/api/action',{action:a})}
function talk(a){const t=document.getElementById('text').value.trim();if(!t)return msg('Напиши текст');post('/api/action',{action:a,text:t})}
const R=(k,v)=>`<div class="row"><span>${k}</span><span>${v}</span></div>`;
async function refresh(){try{const s=await (await fetch('/api/status')).json();
 document.getElementById('dot').className='dot '+(s.running?'on':'off');
 document.getElementById('net').textContent='интернет: '+s.internet;
 let h=R('Айсана',s.running?'работает':'выключена');
 const a=s.aisana;
 if(a&&a.ok){h+=R('OpenAI',a.connected?'на связи':'подключается')+R('Руки',a.arms)+R('Ноги',a.legs)+R('Камера',a.camera)+R('Лиц в памяти',a.faces)+R('Язык',a.language)+R('Юмор',a.humor+'%')+R('Сейчас',a.moving?'двигается':(a.speaking?'говорит':'слушает'))}
 else if(s.running){h+=R('Статус',(a&&a.error)||'запускается…')}
 document.getElementById('state').innerHTML=h;
 const log=document.getElementById('log');const atEnd=log.scrollTop+log.clientHeight>=log.scrollHeight-20;
 log.textContent=(s.log||[]).join('\n');if(atEnd)log.scrollTop=log.scrollHeight;
}catch(e){document.getElementById('dot').className='dot off';msg('Нет связи с роботом')}}
refresh();setInterval(refresh,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, code, body, kind='application/json; charset=utf-8'):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            return self.reply(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
        if self.path == '/api/status':
            return self.reply(200, status())
        self.reply(404, {'ok': False, 'error': 'not found'})

    def do_POST(self):
        length = min(int(self.headers.get('Content-Length') or 0), 8192)
        try:
            body = json.loads(self.rfile.read(length).decode('utf-8') or '{}')
        except ValueError:
            return self.reply(400, {'ok': False, 'error': 'bad json'})
        if self.path == '/api/start':
            return self.reply(200, start())
        if self.path == '/api/stop':
            return self.reply(200, stop())
        if self.path == '/api/action':
            action = body.get('action')
            if action not in aisana_control.ACTIONS or action == 'status':
                return self.reply(400, {'ok': False, 'error': 'unknown action'})
            if not running():
                return self.reply(200, {'ok': False, 'error': 'Айсана выключена'})
            result = aisana_control.send({'action': action, 'text': body.get('text', '')})
            return self.reply(200, result or {'ok': False, 'error': 'Айсана не отвечает'})
        self.reply(404, {'ok': False, 'error': 'not found'})


class Server(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    server = Server(('0.0.0.0', PORT), Handler)
    print('Aisana panel: http://<jetson>:{}  (Ctrl+C to stop)'.format(PORT))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())
