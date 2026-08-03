# AGENTS.md — projekt-emlékeztető a microgpt ágensei számára

Ez a fájl az opencode ágenseinek szóló „memória". Röviden összefoglalja, mi mire
való, és milyen szabályokat érdemes betartani a projekten dolgozva.

## Mi ez a projekt?

Egy **oktató jellegű mini-GPT** (keresztnév-generáló), tiszta Python + numpy,
FastAPI webes felülettel és Plotly vizualizációkkal. Eredete: Karpathy microgpt
— ez a projekt a Value-alapú autogradot vektorizált, kézi numpy backward-ra
cseréli, és böngészős felületet kap. Monorepo: `backend/` (Python) + `frontend/`
(statikus web), docker-compose-zal.

## Fájlok és szerepük

| Fájl | Szerep |
|---|---|
| `backend/microgpt.py` | A mag: adathalmaz, tokenizer, numpy forward/backward, Adam, `train()`, `generate()`, `configure()`, `load_weights()`, `MODEL_LOCK`. |
| `backend/name_ui.py` | FastAPI backend: API + statikus frontend kiszolgálás, újratanítás (háttér-szál), élő loss/PCA, `model.pkl` cache. |
| `backend/viz.py` | Plotly figure-k: loss, rétegszám-összehasonlítás, PCA, attention, eloszlás. |
| `backend/complete_name.py` | Parancssori név-kiegészítő (minden futáskor újratanít, nincs cache). |
| `backend/server.py` | Docker-indító (uvicorn, 8000). |
| `frontend/` | Statikus HTML/JS/CSS (index.html, viz.html, app.js, viz.js, style.css) + nginx. |
| `tests/` | Pytest tesztek: `test_microgpt.py` (mag + referencia-gradiens), `test_viz.py`. |
| `docker-compose.yml` | backend (uvicorn) + frontend (nginx). |
| `README.md` | Bő, felhasználó-orientált dokumentáció. |
| `PLAN.md` | Fejlesztési terv / mérföldkövek. |

## Parancsok (a `backend/` mappában)

- Szerver futtatása: `uv run python name_ui.py`
- Tesztek: `uv run pytest`
- Lint: `uv run ruff check .`
- Új függőség: `uv add <csomag>` (dev: `uv add --dev <csomag>`)
- Docker: a repó gyökeréből `docker compose up --build`
- **GPU-s gépen:** `uv sync --extra gpu` (CuPy; CPU-n/CI-n a sima `uv sync` numpy-fallbackot ad)

## Munkaflow (fontos!)

- **Soha ne commitolj közvetlenül `main`-re.** Minden változtatás **feature
  branch-en** készül (pl. `feat/xyz` vagy `ci/xyz`), majd **PR** megy a `main`-re.
- A CI (`/pr` futtatás) a PR-en fut: `ruff` + `pytest` (3.12/3.13). A GHCR
  image-build csak `main` push után fut.
- PR előtt futtasd lokálisan: `cd backend && uv run ruff check . && uv run pytest -q`.

## Konvenciók és fontos részletek

- **A kód-dokumentáció és a kommentek MAGYARUL vannak** (változónevek angolul).
- A publikus API nevek stabilak: `train(callback=None)`, `generate(prefix, temperature, max_len)`,
  `forward(tokens)`, `backward(cache, grad_logits, gradients)`, `configure(layers, embd, heads, block, fresh)`,
  `load_weights(state, layers, embd, heads, block)`, `reset_params()`, `state_dict`, `param_keys`.
- **Tesztekben SOHA ne importáld a `name_ui`-t** — a modulimport cache-betöltést /
  háttér-tanítást indíthat (amely lefoglalja a `MODEL_LOCK`-ot). Tesztelj
  `microgpt` + `viz` szinten.
- A `model.pkl` cache formátuma: `{n_layer, n_embd, n_head, block_size, state_dict}`.
- A plotly.js-t a szerver **helyben** szolgálja ki (`/plotly.min.js`), CDN nélkül —
  ne cseréld vissza CDN-re.
- A `configure()`-nél `n_embd % n_head == 0` kötelező (különben ValueError).
- A frontend fájlok a `frontend/` mappában vannak; a backend `../frontend`-ből
  olvassa őket. Dockerben a frontend-konténer (nginx) adja ki őket, és az
  API-t proxy-zza a backendre (`nginx.conf`).
- **GPU/CuPy:** az ALAPÉRTELMEZÉS mindig CPU (numpy) — ilyen apró modellnél a
  GPU a kernel-overhead miatt ~20x lassabb. A GPU-t kifejezetten kell bekapcsolni:
  `MICROGPT_BACKEND=gpu` env-var (és a `gpu` extra). `microgpt.BACKEND` =
  'gpu'|'cpu'; létezik `microgpt.set_backend('cpu'|'gpu')` és `microgpt._to_cpu(x)`.
  A tesztek a `conftest.py` autouse-fixture-jével **mindig CPU-n** futnak —
  ne függj a GPU-tól.
- **Cache-portabilitás:** a `model.pkl` mindig CPU-numpy formátumú (a
  `name_ui._save_cache` `_to_cpu`-val ment), a `load_weights` az aktív
  backendre konvertál.
- **A `viz` és az UI CPU-hídat használ** (`viz._cpu`) — a plotly/PCA CPU-t
  igényel, a tanítás lehet GPU-n. A `generate()` `random.choice`-ja
  `size=1`-gyel hívandó (a cupy nem támogatja a size nélküli formát).

## Ismert hibák / figyelmeztetések

- Régi (Value-alapú) `model.pkl` betöltése hibát dob → törölni kell.
- A `train()` (és a viz eldobható tréningjei) a `MODEL_LOCK` alatt futnak —
  tréning közben a `/plot/loss` és `/plot/nlayer` 409-et adnak (a galéria
  automatikusan újrapróbálja).
- Az `input.txt` hiánya esetén a `microgpt` import automatikusan letölti (internet).
- A viz `_snapshot`/`_restore` a TELJES konfigurációt menti/visszaállítja
  (n_layer, n_embd, n_head, block_size, shapes, param_keys, ...) — ha bővíted,
  mindkettőt frissítsd.
- `complete_name.py` minden futáskor újratanít (nincs cache) — a UI a cache-elt
  verziót használja.
