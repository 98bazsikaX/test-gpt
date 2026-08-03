"""
=============================================================================
 microgpt webes felület (FastAPI) — a "backend" része
=============================================================================

Böngészős felület a microgpt modellhez:
  - név-kiegészítés (állítható temperature-rel),
  - újratanítás a böngészőből (n_layer / n_embd / n_head / block_size /
    steps állítható), háttér-szálon fut, és *közben* élőben mutatja a
    loss-görbét és a karakter-embeddingek PCA-ját,
  - vizualizáció-galería (`/viz`): rétegszám-összehasonlítás, PCA,
    attention hőtérképek, következő-betű eloszlás.

A felület HTML/JS/CSS fájljai a `../frontend/` mappában vannak (monorepo
elrendezés, a my-little-jpa mintájára). Helyi futtatáskor ez a backend
szolgálja ki; docker-ben a frontend konténer (nginx) adja ki őket, és a
API-kéréseket proxy-zza a backendre.

Helyi futtatás:  cd backend && uv run python name_ui.py
Docker:          docker compose up (ld. README)
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
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import microgpt
import viz

# -----------------------------------------------------------------------------
# Alapbeállítások
# -----------------------------------------------------------------------------
TEMPERATURE = 0.5   # a mintavételezés "kreativitása" (0.01..5)
N_COMPLETIONS = 5   # hány kiegészítést adjon a /complete
CACHE_FILE = "model.pkl"

# A frontend (HTML/JS/CSS) a monorepo frontend/ mappájában lakik
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))

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


# -----------------------------------------------------------------------------
# Frontend (statikus fájlok a frontend/ mappából)
# -----------------------------------------------------------------------------
def _frontend_file(name: str):
    """Egy frontend fájlt szolgál ki; 404, ha nem található (pl. docker-ben a
    frontend konténer adja ki)."""
    path = os.path.join(FRONTEND_DIR, name)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse({'error': f'hiányzó frontend fájl: {name}'}, status_code=404)


@app.get('/')
def index():
    """A főoldal (frontend/index.html)."""
    return _frontend_file('index.html')


@app.get('/viz')
def viz_page():
    """A vizualizáció-galería oldala (frontend/viz.html)."""
    return _frontend_file('viz.html')


@app.get('/app.js')
def app_js():
    return _frontend_file('app.js')


@app.get('/viz.js')
def viz_js():
    return _frontend_file('viz.js')


@app.get('/style.css')
def style_css():
    return _frontend_file('style.css')


@app.get('/plotly.min.js')
def plotly_js():
    """A plotly.js helyi kiszolgálása (CDN/internet nélkül is megjelennek az ábrák)."""
    return FileResponse(PLOTLY_JS, media_type='application/javascript')


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Indítás (helyi): szabad port + böngésző megnyitása
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
