"""
=============================================================================
 microgpt webes felület (FastAPI)
=============================================================================

Böngészős felület a microgpt modellhez:
  - név-kiegészítés (állítható temperature-rel),
  - újratanítás a böngészőből (n_layer / n_embd / n_head / block_size /
    steps állítható), háttér-szálon fut, és *közben* élőben mutatja a
    loss-görbét és a karakter-embeddingek PCA-ját,
  - vizualizáció-galería (`/viz`): rétegszám-összehasonlítás, PCA,
    attention hőtérképek, következő-betű eloszlás.

Futtatás:  uv run python name_ui.py
"""

import json
import os
import pickle
import socket
import threading
import time
import webbrowser

import plotly
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

import microgpt
import viz

# -----------------------------------------------------------------------------
# Alapbeállítások
# -----------------------------------------------------------------------------
TEMPERATURE = 0.5   # a mintavételezés "kreativitása" (0.01..5)
N_COMPLETIONS = 5   # hány kiegészítést adjon a /complete
CACHE_FILE = "model.pkl"

# A plotly.js-t a telepített plotly csomagból szolgáljuk ki helyben, így a
# böngészőbeli ábrák CDN és internet nélkül is megjelennek.
PLOTLY_JS = os.path.join(os.path.dirname(plotly.__file__), "package_data", "plotly.min.js")


# -----------------------------------------------------------------------------
# A betanított modell cache-kezelése
# -----------------------------------------------------------------------------
def _save_cache():
    """Elmenti a modellt és a hozzá tartozó konfigurációt a `model.pkl`-be."""
    data = {
        'n_layer': microgpt.n_layer,
        'n_embd': microgpt.n_embd,
        'n_head': microgpt.n_head,
        'block_size': microgpt.block_size,
        'state_dict': microgpt.state_dict,
    }
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)


def _load_cache():
    """Betölti a cache-t; ha nincs, háttérben elkezdi a tanítást.

    A régi (csak súlyokat tartalmazó) cache-formátumot figyelmen kívül
    hagyjuk, mert nem tudjuk hozzá a konfigurációt — ilyenkor újratanítunk.
    """
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, dict) and 'state_dict' in data:
            microgpt.load_weights(
                data['state_dict'], data['n_layer'],
                data.get('n_embd', 16), data.get('n_head', 4),
                data.get('block_size', 16))
            return
    start_training({
        'n_layer': microgpt.n_layer,
        'n_embd': microgpt.n_embd,
        'n_head': microgpt.n_head,
        'block_size': microgpt.block_size,
        'steps': microgpt.num_steps,
    })


# -----------------------------------------------------------------------------
# Háttér-szálon futó tanítási feladat (a felület nem akad meg közben)
# -----------------------------------------------------------------------------
_job = {
    'running': False,
    'step': 0,
    'total': 0,
    'losses': [],
    'config': {},
    'error': None,
    'final_loss': None,
    'elapsed': None,
}


def start_training(config):
    """Elindít egy háttér-tanítást. False, ha már fut egy másik."""
    if _job['running']:
        return False
    _job.update(running=True, step=0, total=config['steps'], losses=[],
                config=config, error=None, final_loss=None, elapsed=None)
    _plot_cache.clear()
    threading.Thread(target=_run_job, args=(config,), daemon=True).start()
    return True


def _run_job(config):
    """A tényleges tanítás: új konfiguráció, friss paraméterek, cache-mentés."""
    def on_progress(step, loss):
        _job['step'] = step
        _job['losses'].append(float(loss))

    try:
        with microgpt.MODEL_LOCK:
            microgpt.configure(layers=config['n_layer'], embd=config['n_embd'],
                               heads=config['n_head'], block=config['block_size'],
                               fresh=True)
            microgpt.num_steps = config['steps']
            start = time.time()
            microgpt.train(callback=on_progress)
            _job['elapsed'] = time.time() - start
            _job['final_loss'] = float(_job['losses'][-1]) if _job['losses'] else None
            _save_cache()
    except Exception as e:  # bármi baj van, jelezzük a felületen
        _job['error'] = str(e)
    finally:
        _job['running'] = False


def _validate_config(cfg):
    """A tanítási konfiguráció ellenőrzése; hibaüzenetet vagy None-t ad vissza."""
    if cfg['n_layer'] < 1 or cfg['n_layer'] > 8:
        return 'n_layer 1..8 között legyen'
    if cfg['n_embd'] < 4 or cfg['n_embd'] > 256:
        return 'n_embd 4..256 között legyen'
    if cfg['n_head'] < 1 or cfg['n_head'] > 16:
        return 'n_head 1..16 között legyen'
    if cfg['n_embd'] % cfg['n_head'] != 0:
        return 'n_embd osztható legyen n_head-dal'
    if cfg['block_size'] < 2 or cfg['block_size'] > 64:
        return 'block_size 2..64 között legyen'
    if cfg['steps'] < 10 or cfg['steps'] > 200000:
        return 'steps 10..200000 között legyen'
    return None


# -----------------------------------------------------------------------------
# FastAPI alkalmazás
# -----------------------------------------------------------------------------
app = FastAPI(title='microgpt', description='Mini GPT: kiegészítés, tanítás, vizualizációk.')


class CompleteRequest(BaseModel):
    seed: str


class TemperatureRequest(BaseModel):
    value: float


class TrainRequest(BaseModel):
    n_layer: int = 4
    n_embd: int = 16
    n_head: int = 4
    block_size: int = 16
    steps: int = 32000


@app.get('/', response_class=HTMLResponse)
def index():
    """A főoldal: kiegészítés + tanítás + élő vizualizációk."""
    return HTMLResponse(PAGE)


@app.post('/complete')
def complete(request: CompleteRequest):
    """A megadott előtag kiegészítése `N_COMPLETIONS` névvel."""
    seed = request.seed.lower()
    seed = ''.join(c for c in seed if c in microgpt.uchars)
    if not seed:
        return JSONResponse({'results': []})
    results = [microgpt.generate(seed, TEMPERATURE) for _ in range(N_COMPLETIONS)]
    return JSONResponse({'results': results, 'temperature': TEMPERATURE})


@app.post('/temperature')
def set_temperature(request: TemperatureRequest):
    """A mintavételezési hőmérséklet beállítása (0.01..5)."""
    global TEMPERATURE
    if not (0.01 <= request.value <= 5.0):
        return JSONResponse({'error': '0.01..5.0 között legyen'}, status_code=400)
    TEMPERATURE = request.value
    return JSONResponse({'temperature': TEMPERATURE})


@app.get('/config')
def get_config():
    """Az aktuális konfiguráció (a form kitöltéséhez)."""
    return JSONResponse({
        'n_layer': microgpt.n_layer, 'n_embd': microgpt.n_embd,
        'n_head': microgpt.n_head, 'block_size': microgpt.block_size,
        'steps': microgpt.num_steps, 'temperature': TEMPERATURE,
        'training': _job['running'],
    })


@app.post('/retrain')
def retrain(request: TrainRequest):
    """Új tanítás indítása a megadott hiperparaméterekkel."""
    cfg = {
        'n_layer': request.n_layer, 'n_embd': request.n_embd,
        'n_head': request.n_head, 'block_size': request.block_size,
        'steps': request.steps,
    }
    error = _validate_config(cfg)
    if error:
        return JSONResponse({'error': error}, status_code=400)
    if not start_training(cfg):
        return JSONResponse({'error': 'már fut egy tanítás'}, status_code=409)
    return JSONResponse({'ok': True})


# -----------------------------------------------------------------------------
# Vizualizációk: élő tanítás + galéria
# -----------------------------------------------------------------------------
@app.get('/plot/live')
def plot_live():
    """A futó/háttér-tanítás állapota: loss-görbe + élő embedding-PCA."""
    data = {
        'running': _job['running'],
        'step': _job['step'],
        'total': _job['total'],
        'losses': list(_job['losses']),
        'final_loss': _job['final_loss'],
        'elapsed': _job['elapsed'],
        'error': _job['error'],
        'config': _job['config'],
    }
    try:  # a karakter-embeddingek aktuális PCA-ja (a tanulás közben "mozog")
        emb = microgpt.state_dict['token_embedding']
        proj = viz._pca(emb)
        data['pca'] = proj.tolist()
        data['pca_labels'] = microgpt.uchars + ['BOS']
    except Exception:
        pass
    return JSONResponse(data)


_plot_cache = {}


def _cached_fig(key, fn):
    """Egy Plotly figure JSON-ját egyszer számolja ki, utána cache-eli."""
    if key not in _plot_cache:
        _plot_cache[key] = json.loads(fn().to_json())
    return _plot_cache[key]


@app.get('/plot/loss')
def plot_loss():
    """A tréning loss-görbéje (eldobható modellen, a jelenlegi konfigurációval)."""
    if _job['running']:
        return JSONResponse({'error': 'tanítás fut, várj'}, status_code=409)
    return JSONResponse(_cached_fig('loss', viz.loss_fig))


@app.get('/plot/nlayer')
def plot_nlayer():
    """1/2/4 réteg loss-összehasonlítása."""
    if _job['running']:
        return JSONResponse({'error': 'tanítás fut, várj'}, status_code=409)
    return JSONResponse(_cached_fig('nlayer', viz.nlayer_fig))


@app.get('/plot/pca_token')
def plot_pca_token():
    return JSONResponse(_cached_fig('pca_token', viz.pca_token_fig))


@app.get('/plot/pca_pos')
def plot_pca_pos():
    return JSONResponse(_cached_fig('pca_pos', viz.pca_position_fig))


@app.get('/plot/attention')
def plot_attention(name: str = 'karla'):
    """Attention hőtérkép a megadott névre."""
    try:
        fig = viz.attention_fig(name)
    except ValueError as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    return JSONResponse(json.loads(fig.to_json()))


@app.get('/plot/distribution')
def plot_distribution(prefix: str = 'ka'):
    """A következő betű eloszlása az adott prefix után."""
    try:
        fig = viz.distribution_fig(prefix)
    except ValueError as e:
        return JSONResponse({'error': str(e)}, status_code=400)
    return JSONResponse(json.loads(fig.to_json()))


@app.get('/viz', response_class=HTMLResponse)
def viz_page():
    """A vizualizáció-galería oldala."""
    return HTMLResponse(VIZ_PAGE)


@app.get('/plotly.min.js')
def plotly_js():
    """A plotly.js helyi kiszolgálása (CDN/internet nélkül is megjelennek az ábrák)."""
    return FileResponse(PLOTLY_JS, media_type='application/javascript')


# -----------------------------------------------------------------------------
# A főoldal HTML-je
# -----------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="hu">
<head>
<meta charset="utf-8">
<title>microgpt</title>
<script src="/plotly.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 860px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 24px; }
  h2 { font-size: 18px; margin-top: 28px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  input[type=text], input[type=number] { font-size: 15px; padding: 6px 8px; border: 1px solid #999; border-radius: 4px; }
  input[type=range] { width: 180px; vertical-align: middle; }
  button { font-size: 15px; padding: 7px 14px; border: 1px solid #999; border-radius: 4px; background: #eee; cursor: pointer; }
  ul { list-style: none; padding: 0; }
  li { font-size: 20px; padding: 5px 0; border-bottom: 1px solid #eee; }
  #err, #trainstatus { color: #b00; font-size: 14px; }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin: 6px 0; }
  label { font-size: 14px; }
  a { color: #06c; }
  .chart { min-height: 330px; }
</style>
</head>
<body>
<h1>microgpt</h1>

<h2>Név-kiegészítés</h2>
<div class="row">
  <input id="seed" type="text" placeholder="pl. ka" autocomplete="off">
  <button onclick="doComplete()">Kiegészítés</button>
  <label>temperature
    <input id="temp" type="range" min="0.05" max="2" step="0.05" value="0.5" oninput="setTemp()">
    <span id="tempval">0.50</span>
  </label>
</div>
<ul id="out"></ul>
<p id="err"></p>

<h2>Újratanítás</h2>
<div class="row">
  <label>rétegek <input id="f_layers" type="number" min="1" max="8"></label>
  <label>embd <input id="f_embd" type="number" min="4" max="256"></label>
  <label>fejek <input id="f_heads" type="number" min="1" max="16"></label>
  <label>block <input id="f_block" type="number" min="2" max="64"></label>
  <label>steps <input id="f_steps" type="number" min="10" max="200000"></label>
  <button onclick="startTrain()">Tanítás</button>
</div>
<p id="trainstatus"></p>
<div id="live_loss" class="chart"></div>
<div id="live_pca" class="chart"></div>

<p><a href="/viz">Vizualizációk galéria &rarr;</a></p>

<script>
const lossLayout = { title: 'Élő loss', xaxis: {title:'lépés'}, yaxis: {title:'loss'}, template:'plotly_white', height: 320 };
const pcaLayout  = { title: 'Karakter-embeddingek PCA-ja (élő)', xaxis:{title:'1. főkomponens'}, yaxis:{title:'2. főkomponens'}, template:'plotly_white', height: 320 };

async function api(url, method, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  return res.json();
}

async function doComplete() {
  const out = document.getElementById('out'); out.innerHTML = '';
  const err = document.getElementById('err'); err.textContent = '';
  const seed = document.getElementById('seed').value.trim();
  if (!seed) return;
  const d = await api('/complete', 'POST', { seed });
  if (!d.results || !d.results.length) { err.textContent = 'Nincs érvényes betű (a-z).'; return; }
  for (const n of d.results) { const li = document.createElement('li'); li.textContent = n; out.appendChild(li); }
}

async function setTemp() {
  const t = parseFloat(document.getElementById('temp').value);
  document.getElementById('tempval').textContent = t.toFixed(2);
  await api('/temperature', 'POST', { value: t });
}

async function startTrain() {
  const cfg = {
    n_layer: +document.getElementById('f_layers').value,
    n_embd: +document.getElementById('f_embd').value,
    n_head: +document.getElementById('f_heads').value,
    block_size: +document.getElementById('f_block').value,
    steps: +document.getElementById('f_steps').value,
  };
  const d = await api('/retrain', 'POST', cfg);
  if (d.error) { document.getElementById('trainstatus').textContent = 'Hiba: ' + d.error; return; }
  document.getElementById('trainstatus').textContent = 'Tanítás indul...';
  pollLive();
}

let polling = false;
async function pollLive() {
  if (polling) return;
  polling = true;
  try {
    const d = await api('/plot/live', 'GET');
    const st = document.getElementById('trainstatus');
    if (d.losses && d.losses.length) {
      const x = Array.from({ length: d.losses.length }, (_, i) => i + 1);
      Plotly.react('live_loss', [{ x, y: d.losses, mode: 'lines', name: 'loss' }], lossLayout);
    }
    if (d.pca && d.pca.length) {
      Plotly.react('live_pca', [{
        x: d.pca.map(p => p[0]), y: d.pca.map(p => p[1]),
        text: d.pca_labels, mode: 'markers+text', textposition: 'top center',
      }], pcaLayout);
    }
    if (d.running) {
      st.textContent = 'Tanulás... ' + d.step + '/' + d.total;
      setTimeout(pollLive, 500);
    } else if (d.error) {
      st.textContent = 'Hiba: ' + d.error;
    } else if (d.losses && d.losses.length) {
      st.textContent = 'Kész! ' + d.total + ' lépés, utolsó loss: ' + d.final_loss.toFixed(4);
    }
  } finally { polling = false; }
}

async function loadConfig() {
  const d = await api('/config', 'GET');
  document.getElementById('f_layers').value = d.n_layer;
  document.getElementById('f_embd').value = d.n_embd;
  document.getElementById('f_heads').value = d.n_head;
  document.getElementById('f_block').value = d.block_size;
  document.getElementById('f_steps').value = d.steps;
  document.getElementById('temp').value = d.temperature;
  document.getElementById('tempval').textContent = d.temperature.toFixed(2);
  if (d.training) pollLive();
}

document.getElementById('seed').addEventListener('keydown', e => { if (e.key === 'Enter') doComplete(); });
loadConfig();
pollLive();
</script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# A /viz galéria HTML-je
# -----------------------------------------------------------------------------
VIZ_PAGE = """<!doctype html>
<html lang="hu">
<head>
<meta charset="utf-8">
<title>microgpt vizualizációk</title>
<script src="/plotly.min.js"></script>
<style>
  body { font-family: system-ui, sans-serif; max-width: 900px; margin: 24px auto; padding: 0 16px; color: #222; }
  h1 { font-size: 22px; }
  h2 { font-size: 17px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
  input { font-size: 15px; padding: 6px 8px; border: 1px solid #999; border-radius: 4px; }
  button { font-size: 15px; padding: 7px 14px; border: 1px solid #999; border-radius: 4px; background: #eee; cursor: pointer; }
  .row { margin: 8px 0; }
  .chart { min-height: 380px; }
</style>
</head>
<body>
<h1><a href="/">&larr; vissza</a> microgpt vizualizációk</h1>

<h2>Rétegszám-összehasonlítás</h2>
<div id="pnlayer" class="chart"></div>

<h2>Pozíció-embeddingek PCA-ja</h2>
<div id="ppca_pos" class="chart"></div>

<h2>Attention hőtérkép</h2>
<div class="row">
  <input id="att_name" type="text" value="karla">
  <button onclick="att()">Rajzol</button>
</div>
<div id="patt" class="chart"></div>

<h2>Következő betű eloszlása</h2>
<div class="row">
  <input id="dist_prefix" type="text" value="ka">
  <button onclick="dist()">Rajzol</button>
</div>
<div id="pdist" class="chart"></div>

<script>
async function loadPlot(url, id, tries) {
  tries = tries || 0;
  const res = await fetch(url);
  if (res.status === 409 && tries < 30) {   // tréning fut -> várunk és újrapróbáljuk
    setTimeout(() => loadPlot(url, id, tries + 1), 2000);
    return;
  }
  const d = await res.json();
  const el = document.getElementById(id);
  if (d.error) { el.innerHTML = '<p style="color:#b00">' + d.error + '</p>'; return; }
  Plotly.newPlot(el, d.data, d.layout, { responsive: true });
}
function att()  { loadPlot('/plot/attention?name=' + encodeURIComponent(document.getElementById('att_name').value), 'patt'); }
function dist() { loadPlot('/plot/distribution?prefix=' + encodeURIComponent(document.getElementById('dist_prefix').value), 'pdist'); }
loadPlot('/plot/nlayer', 'pnlayer');
loadPlot('/plot/pca_pos', 'ppca_pos');
att();
dist();
</script>
</body>
</html>
"""


# -----------------------------------------------------------------------------
# Indítás: szabad port + böngésző megnyitása
# -----------------------------------------------------------------------------
def _pick_free_port() -> int:
    """Egy szabad (még lefoglalatlan) TCP portot kér az operációs rendszertől."""
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == '__main__':
    _load_cache()  # cache betöltése, vagy háttér-tanítás indítása
    port = _pick_free_port()
    url = f'http://127.0.0.1:{port}'
    print(f'open {url} in your browser')
    # Kis késleltetéssel nyissuk meg a böngészőt, hogy a szerver előbb elinduljon
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host='127.0.0.1', port=port, log_level='warning')
