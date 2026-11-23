"""Phone control panel for Aisana:  python3 aisana_panel.py   ->  http://<jetson>:8080

Buttons: start / stop Aisana, emergency STOP, wave, dance, iPhone microphone (push to talk
or always on, https only), speaker volume, humor slider, "say this", "ask her",
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
NOISE = re.compile(r'Mic level|GST_ARGUS|CONSUMER|Exposure Range|Camera (?:index|mode)|Output Stream|seconds to Run|'
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
    if name == 'wlan0':
        try:
            out = subprocess.check_output(['nmcli', '-t', '-g', 'GENERAL.CONNECTION', 'device', 'show', 'wlan0'],
                                          stderr=subprocess.DEVNULL).decode('utf-8', 'replace').strip()
        except (OSError, subprocess.CalledProcessError):
            out = ''
        return 'Wi-Fi ' + out if out else 'Wi-Fi'
    return {'eth1': 'роутер 4G Altel (USB)'}.get(name, name)


SPEAKER_LABEL = os.environ.get('AISANA_SPEAKER_LABEL', 'USB Composite Device')   # Xiaomi
SPEAKER_CONTROL = os.environ.get('AISANA_SPEAKER_CONTROL', 'PCM')
VOLUME_FILE = os.environ.get('AISANA_VOLUME_FILE', os.path.expanduser('~/.local/share/aisana/volume.txt'))
_last_card = {'card': None}


def speaker_card(cards_text=None):
    """ALSA card number of the speaker, found by its label every time (the number
    changes after a USB reconnect). None if it is not plugged in: no fallback."""
    if cards_text is None:
        try:
            with open('/proc/asound/cards') as handle:
                cards_text = handle.read()
        except (IOError, OSError):
            return None
    for line in cards_text.splitlines():
        found = re.match(r'\s*(\d+)\s+\[[^\]]*\]:\s.*-\s*(.+?)\s*$', line)
        if found and found.group(2) == SPEAKER_LABEL:
            return int(found.group(1))
    return None


def parse_volume(amixer_text):
    found = re.search(r'\[(\d+)%\]', amixer_text)
    return int(found.group(1)) if found else None


def get_volume(card):
    try:
        out = subprocess.check_output(['amixer', '-c', str(card), 'sget', SPEAKER_CONTROL],
                                      stderr=subprocess.DEVNULL).decode('utf-8', 'replace')
    except (OSError, subprocess.CalledProcessError):
        return None
    return parse_volume(out)


def saved_volume():
    try:
        with open(VOLUME_FILE) as handle:
            return max(0, min(100, int(handle.read().strip())))
    except (IOError, OSError, ValueError):
        return None


def set_volume(value, remember=True):
    card = speaker_card()
    if card is None:
        return {'ok': False, 'error': 'колонка Xiaomi не подключена'}
    value = max(0, min(100, int(value)))
    code = subprocess.call(['amixer', '-q', '-c', str(card), 'sset', SPEAKER_CONTROL,
                            '{}%'.format(value), 'unmute'], stderr=subprocess.DEVNULL)
    if code != 0:
        return {'ok': False, 'error': 'amixer не смог поменять громкость'}
    if remember:
        try:
            folder = os.path.dirname(VOLUME_FILE)
            if folder and not os.path.isdir(folder):
                os.makedirs(folder)
            with open(VOLUME_FILE, 'w') as handle:
                handle.write('{}\n'.format(value))
        except (IOError, OSError):
            pass
    return {'ok': True, 'volume': get_volume(card)}


def speaker_volume():
    """Current volume in %, or None. After a replug (new card) the saved level is
    put back, because the speaker comes back at its default level."""
    card = speaker_card()
    if card is None:
        _last_card['card'] = None
        return None
    if card != _last_card['card']:
        _last_card['card'] = card
        level = saved_volume()
        if level is not None:
            set_volume(level, remember=False)
    return get_volume(card)


def status():
    state = {'running': running(), 'internet': internet(), 'log': log_tail(),
             'volume': speaker_volume()}
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
 .grid.h5{grid-template-columns:repeat(5,1fr);gap:6px}.grid.h5 button{font-size:13px;padding:10px 2px}
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
 .sel{outline:3px solid #3b82f6}
 #ptt{display:none;width:100%;margin-top:10px;padding:34px 10px;font-size:22px;background:#7c3aed;
      -webkit-user-select:none;user-select:none;-webkit-touch-callout:none;touch-action:none}
 #ptt.down{background:#3ecf6e;color:#062}
 #lvl{height:8px;border-radius:4px;background:#232733;margin-top:10px;overflow:hidden}
 #lvl div{height:100%;width:0;background:#3ecf6e;transition:width .1s}
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
 <div class="card">
  <div class="row"><span>🎤 Микрофон iPhone</span><b id="mv">выключен</b></div>
  <div class="grid">
   <button id="m-ptt" onclick="micStart('ptt')">Режим кнопки</button>
   <button id="m-live" onclick="micStart('live')">Слушать всё время</button>
  </div>
  <button id="ptt">Зажми и говори</button>
  <div id="lvl"><div></div></div>
  <div class="grid"><button class="off-btn" style="grid-column:span 2" onclick="micStop('Микрофон iPhone выключен, слушает USB-микрофон')">Выключить микрофон iPhone</button></div>
 </div>
 <div class="card">
  <div class="row"><span>🔊 Громкость колонки</span><b id="vv">—</b></div>
  <input id="vol" type="range" min="0" max="100" step="5" style="width:100%"
   oninput="vTouch=Date.now();$('vv').textContent=this.value+'%'"
   onchange="setV(this.value)">
  <div class="grid h5">
   <button onclick="setV(30)">30</button><button onclick="setV(50)">50</button><button onclick="setV(70)">70</button><button onclick="setV(85)">85</button><button onclick="setV(100)">100</button>
  </div>
  <div class="grid"><button style="grid-column:span 2" onclick="testV()">Проверить звук</button></div>
 </div>
 <div class="card">
  <div class="row"><span>😏 Юмор и сарказм</span><b id="hv">—</b></div>
  <input id="humor" type="range" min="0" max="100" step="5" style="width:100%"
   oninput="hTouch=Date.now();document.getElementById('hv').textContent=this.value+'%'"
   onchange="hTouch=Date.now();post('/api/action',{action:'humor',text:this.value})">
  <div class="grid h5">
   <button onclick="setH(0)">0 серьёзно</button><button onclick="setH(25)">25</button><button onclick="setH(45)">45</button><button onclick="setH(75)">75</button><button onclick="setH(100)">100 TARS</button>
  </div>
 </div>
 <div id="msg"></div>
 <div class="card" id="state"></div>
 <div class="card"><pre id="log"></pre></div>
</main>
<script>
const msg=t=>{document.getElementById('msg').textContent=t};
async function post(url,body){try{const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});const j=await r.json();msg(j.ok?'Готово':'Ошибка: '+(j.error||''));setTimeout(refresh,800)}catch(e){msg('Нет связи с роботом')}}
// iPhone microphone: 24 kHz mono 16-bit PCM to /api/mic, one request at a time.
let mic=null,pressed=false,sending=false,lastSent=0,parts=[],wake=null;
const $=id=>document.getElementById(id);
function micShow(){$('m-ptt').className=mic&&mic.mode==='ptt'?'sel':'';$('m-live').className=mic&&mic.mode==='live'?'sel':'';
 $('ptt').style.display=mic&&mic.mode==='ptt'?'block':'none';
 $('mv').textContent=!mic?'выключен':(mic.mode==='ptt'?(pressed?'слушает…':'кнопка'):'слушает всё время')}
async function micStart(mode){
 if(!window.isSecureContext||!navigator.mediaDevices||!navigator.mediaDevices.getUserMedia)
  return msg('Микрофон работает только по https-ссылке: https://mmae603-desktop.tail16cc45.ts.net');
 if(mic){mic.mode=mode;pressed=false;micShow();return}
 let stream;
 try{stream=await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:true,noiseSuppression:true,autoGainControl:true,channelCount:1}})}
 catch(e){return msg('Нет доступа к микрофону: разреши его в Safari (аА → Настройки сайта)')}
 const AC=window.AudioContext||window.webkitAudioContext;const ctx=new AC();
 try{await ctx.resume()}catch(e){}
 const src=ctx.createMediaStreamSource(stream);const node=ctx.createScriptProcessor(4096,1,1);
 const ratio=ctx.sampleRate/24000;let prev=0,pos=0;
 node.onaudioprocess=e=>{
  const inp=e.inputBuffer.getChannelData(0),n=inp.length;
  let peak=0;for(let i=0;i<n;i+=16){const v=Math.abs(inp[i]);if(v>peak)peak=v}
  $('lvl').firstChild.style.width=Math.min(100,peak*140)+'%';
  if(!mic||(mic.mode==='ptt'&&!pressed)){prev=inp[n-1];pos=0;return}
  const out=new Int16Array(Math.ceil((n-pos)/ratio)+1);let k=0,t=pos;
  while(t<n){const i=Math.floor(t),f=t-i,a=i===0?prev:inp[i-1],b=inp[i];
   let v=a+(b-a)*f;v=v<-1?-1:v>1?1:v;out[k++]=v*32767;t+=ratio}
  pos=t-n;prev=inp[n-1];parts.push(out.subarray(0,k))};
 src.connect(node);node.connect(ctx.destination);
 mic={mode,stream,ctx,node,src};pressed=false;micShow();
 try{if(navigator.wakeLock)wake=await navigator.wakeLock.request('screen')}catch(e){}
 msg(mode==='ptt'?'Зажми кнопку и говори, имя «Айсана» не нужно':'iPhone слушает вместо USB-микрофона; обращайся «Айсана, …»; не блокируй экран')}
function micStop(note){if(!mic)return;try{mic.node.disconnect();mic.src.disconnect();mic.stream.getTracks().forEach(t=>t.stop());mic.ctx.close()}catch(e){}
 mic=null;pressed=false;parts=[];try{if(wake)wake.release()}catch(e){}wake=null;$('lvl').firstChild.style.width='0';micShow();if(note)msg(note)}
function b64(bytes){let s='';for(let i=0;i<bytes.length;i+=0x8000)s+=String.fromCharCode.apply(null,bytes.subarray(i,i+0x8000));return btoa(s)}
async function micSend(){
 if(!mic||sending)return;
 let len=0;parts.forEach(p=>len+=p.length);
 if(!len&&Date.now()-lastSent<700&&!(mic.wasPressed&&!pressed))return;   // heartbeat keeps the USB mic muted
 const pcm=new Int16Array(len);let o=0;parts.forEach(p=>{pcm.set(p,o);o+=p.length});parts=[];
 const body={mode:mic.mode,pressed:pressed,pcm:b64(new Uint8Array(pcm.buffer))};mic.wasPressed=pressed;
 sending=true;lastSent=Date.now();
 try{const r=await fetch('/api/mic',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j=await r.json();if(!j.ok)msg('Микрофон: '+(j.error||'ошибка'))}catch(e){msg('Микрофон: нет связи с роботом')}
 sending=false}
setInterval(micSend,100);
function down(e){e.preventDefault();if(!mic)return;pressed=true;$('ptt').className='down';$('ptt').textContent='Говори… отпусти, когда закончишь';micShow()}
function up(e){e.preventDefault();if(!pressed)return;pressed=false;$('ptt').className='';$('ptt').textContent='Зажми и говори';micShow();micSend()}
['touchstart','mousedown'].forEach(n=>$('ptt').addEventListener(n,down));
['touchend','touchcancel','mouseup','mouseleave'].forEach(n=>$('ptt').addEventListener(n,up));
document.addEventListener('visibilitychange',()=>{if(document.hidden&&mic)micStop('Экран был выключен: микрофон iPhone остановлен, включи снова')});
function act(a){post('/api/action',{action:a})}
let hTouch=0,vTouch=0;
function usbLevel(a){if(a.muted)return 'не слушает, пока говорит';const v=a.usb_level||0;
 const bars=v<300?'▁':v<800?'▂▃':v<2000?'▂▃▅':'▂▃▅▇';return bars+' '+(v<300?'тишина':v<800?'тихо':v<2000?'слышит голос':'громко')}
async function setV(v){vTouch=Date.now();$('vol').value=v;$('vv').textContent=v+'%';
 try{const j=await (await fetch('/api/volume',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({volume:Number(v)})})).json();
  msg(j.ok?'Громкость '+j.volume+'%':'Ошибка: '+j.error)}catch(e){msg('Нет связи с роботом')}}
function testV(){post('/api/action',{action:'say',text:'Testing my volume. Can you hear me well?'})}
function setH(v){hTouch=Date.now();document.getElementById('humor').value=v;document.getElementById('hv').textContent=v+'%';post('/api/action',{action:'humor',text:String(v)})}
function talk(a){const t=document.getElementById('text').value.trim();if(!t)return msg('Напиши текст');post('/api/action',{action:a,text:t})}
const R=(k,v)=>`<div class="row"><span>${k}</span><span>${v}</span></div>`;
async function refresh(){try{const s=await (await fetch('/api/status')).json();
 document.getElementById('dot').className='dot '+(s.running?'on':'off');
 document.getElementById('net').textContent='интернет: '+s.internet;
 let h=R('Айсана',s.running?'работает':'выключена');
 const a=s.aisana;
 if(a&&a.ok){h+=R('OpenAI',a.connected?'на связи':'подключается')+R('Руки',a.arms)+R('Ноги',a.legs)+R('Камера',a.camera)+R('Лиц в памяти',a.faces)+R('Язык',a.language)+R('Юмор',a.humor+'%')+R('Микрофон',(a.mic||'USB')+(a.mic==='USB'||!a.mic?' · '+usbLevel(a):''))+R('Сейчас',a.moving?'двигается':(a.speaking?'говорит':'слушает'))}
 else if(s.running){h+=R('Статус',(a&&a.error)||'запускается…')}
 document.getElementById('state').innerHTML=h;
 if(Date.now()-vTouch>5000){if(s.volume==null){$('vv').textContent='нет колонки'}else{$('vol').value=s.volume;$('vv').textContent=s.volume+'%'}}
 if(a&&a.ok&&Date.now()-hTouch>5000){document.getElementById('humor').value=a.humor;document.getElementById('hv').textContent=a.humor+'%'}
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
        limit = aisana_control.MAX_LINE if self.path == '/api/mic' else 8192
        length = min(int(self.headers.get('Content-Length') or 0), limit)
        try:
            body = json.loads(self.rfile.read(length).decode('utf-8') or '{}')
        except ValueError:
            return self.reply(400, {'ok': False, 'error': 'bad json'})
        if self.path == '/api/start':
            return self.reply(200, start())
        if self.path == '/api/stop':
            return self.reply(200, stop())
        if self.path == '/api/volume':
            try:
                return self.reply(200, set_volume(int(body.get('volume'))))
            except (TypeError, ValueError):
                return self.reply(400, {'ok': False, 'error': 'volume must be 0-100'})
        if self.path == '/api/mic':
            result = aisana_control.send({'action': 'mic', 'mode': body.get('mode'),
                                          'pressed': bool(body.get('pressed')), 'pcm': body.get('pcm', '')})
            return self.reply(200, result or {'ok': False, 'error': 'Айсана выключена'})
        if self.path == '/api/action':
            action = body.get('action')
            if action not in aisana_control.ACTIONS or action in ('status', 'mic'):
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
