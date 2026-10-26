"""Loopback-only progress view for the current local restoration work."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT/'runs/work_status.json'
HTML = '''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Restoration work · Live progress</title>
<style>body{background:#101419;color:#eef0ed;font:15px system-ui;margin:0;padding:24px;max-width:1000px}h1{font-size:26px}article{background:#1b222b;padding:18px;border-radius:14px;margin:14px 0}p{line-height:1.6}small{color:#aab6bd}progress{width:100%;accent-color:#d5b778}pre{white-space:pre-wrap;font:12px monospace;max-height:220px;overflow:auto}img{width:100%;border-radius:10px}a{color:#d5b778}</style>
<h1>Photo restoration · Live work</h1><small>Local RTX 4050 project · Updates every 3 seconds</small>
<article><h2>Current step</h2><p id="step"></p><small id="note"></small></article>
<section id="assets"></section><article><h2>Training</h2><pre id="training"></pre></article>
<article><h2>Latest verified portrait</h2><small id="previewNote">Epoch 40 baseline. New approach has not been verified yet.</small><img id="preview" src="/portrait.png"></article>
<p><a href="http://127.0.0.1:8501/">Open restoration app</a></p>
<script>async function update(){try{let s=await(await fetch('/status')).json();document.getElementById('step').textContent=s.step;document.getElementById('note').textContent=s.note;document.getElementById('assets').replaceChildren();for(let x of s.assets){let a=document.createElement('article');let h=document.createElement('h2');h.textContent=x.name;a.append(h);let p=document.createElement('p');p.textContent=x.message;a.append(p);let bar=document.createElement('progress');if(x.total){bar.max=x.total;bar.value=x.bytes}a.append(bar);document.getElementById('assets').append(a)}document.getElementById('training').textContent=s.training;if(s.preview_note)document.getElementById('previewNote').textContent=s.preview_note;document.getElementById('preview').src='/portrait.png?v='+s.preview_mtime}catch(e){document.getElementById('note').textContent='Waiting for local status server…'}}update();setInterval(update,3000)</script>'''


def status():
    state = json.loads(STATE.read_text()) if STATE.exists() else {'step':'Preparing research downloads','note':'Existing epoch-40 app remains active.'}
    assets = []
    for label,name,total in [('Learned old-photo detector','data/external/global_checkpoints.zip',2036400762),('Learned inpainting model (LaMa)','weights/lama/big-lama.pt',205706611)]:
        file = ROOT/name
        partial = file.with_suffix(file.suffix+'.part')
        candidate = partial if partial.exists() else file
        size = candidate.stat().st_size if candidate.exists() else 0
        verified = file.with_suffix('.download.json').exists()
        message = f'{size/1e6:.1f} MB received' + (' · checksum verified' if verified else ' · not verified/ready yet')
        assets.append({'name':label,'bytes':size,'total':total,'message':message})
    manifest_file = ROOT/'data/external/celeba_hf_manifest.json'
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding='utf-8-sig'))
        folder = ROOT/'data/external/celeba_shards'
        size = sum(file.stat().st_size for file in folder.glob('*.parquet'))
        for partial in folder.glob('*.parquet.part'):
            progress = partial.with_suffix('').with_suffix('.progress.json')
            size += json.loads(progress.read_text()).get('bytes_received',0) if progress.exists() else partial.stat().st_size
        verified = len(list(folder.glob('*.verified.json')))
        preparation = ROOT/'data/portraits/celeba/preparation_status.json'
        prepared = json.loads(preparation.read_text()).get('prepared',{}) if preparation.exists() else {}
        assets.append({'name':'CelebA portrait dataset · official splits','bytes':size,
                       'total':sum(item['size'] for item in manifest['files']),
                       'message':f'{size/1e9:.2f} GB received · {verified}/25 shards verified · prepared {prepared}'})
    state['assets'] = assets
    log = ROOT/state.get('training_log','runs/long-auto-training.log')
    state['training'] = '\n'.join(log.read_text(errors='replace').splitlines()[-8:]) if log.exists() else 'No new training launched yet.'
    preview = ROOT/state.get('preview','outputs/long_training_report/epoch40_portrait.png')
    state['preview_mtime'] = preview.stat().st_mtime_ns if preview.exists() else 0
    return state


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split('?')[0]
        if path == '/':
            body,kind = HTML.encode(),'text/html; charset=utf-8'
        elif path == '/status':
            body,kind = json.dumps(status()).encode(),'application/json'
        elif path == '/portrait.png':
            state = status()
            preview = (ROOT/state.get('preview','outputs/long_training_report/epoch40_portrait.png')).resolve()
            if not preview.is_relative_to((ROOT/'outputs').resolve()) or not preview.is_file():
                self.send_error(404)
                return
            body,kind = preview.read_bytes(),'image/png'
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header('Content-Type',kind)
        self.send_header('Cache-Control','no-store')
        self.end_headers()
        self.wfile.write(body)


if __name__=='__main__':
    print('Local progress: http://127.0.0.1:8502',flush=True)
    ThreadingHTTPServer(('127.0.0.1',8502),Handler).serve_forever()
