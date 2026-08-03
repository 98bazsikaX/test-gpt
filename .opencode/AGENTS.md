# AGENTS.md — projekt-emlékeztető a microgpt ágensei számára

Ez a fájl az opencode ágenseinek szóló „memória". Röviden összefoglalja, mi mire
való, és milyen szabályokat érdemes betartani a projekten dolgozva.

## Mi ez a projekt?

Egy **oktató jellegű mini-GPT** (keresztnév-generáló), tiszta Python + numpy,
FastAPI webes felülettel és Plotly vizualizációkkal. Eredete: Karpathy microgpt
— ez a projekt a Value-alapú autogradot vektorizált, kézi numpy backward-ra
cseréli, és böngészős felületet kap.

## Fájlok és szerepük

| Fájl | Szerep |
|---|---|
| `microgpt.py` | A mag: adathalmaz, tokenizer, numpy forward/backward, Adam, `train()`, `generate()`, `configure()`, `load_weights()`, `MODEL_LOCK`. |
| `name_ui.py` | FastAPI UI: név-kiegészítés, temperature, újratanítás (háttér-szál), élő loss/PCA, `/viz` galéria, `model.pkl` cache. |
| `viz.py` | Plotly figure-k: loss, rétegszám-összehasonlítás, PCA, attention, eloszlás. |
| `complete_name.py` | Parancssori név-kiegészítő (minden futáskor újratanít, nincs cache). |
| `tests/` | Pytest tesztek: `test_microgpt.py` (mag + referencia-gradiens), `test_viz.py`. |
| `README.md` | Bő, felhasználó-orientált dokumentáció. |

## Parancsok

- Szerver futtatása: `uv run python name_ui.py`
- Tesztek: `uv run pytest`
- Lint: `uv run ruff check .`
- Új függőség: `uv add <csomag>` (dev: `uv add --dev <csomag>`)

## Konvenciók és fontos részletek

- **A kód-dokumentáció és a kommentek MAGYARUL vannak** (változónevek angolul).
- A publikus API nevek stabilak: `train(callback=None)`, `generate(prefix, temperature, max_len)`,
  `forward(tokens)`, `backward(cache, grad_logits, gradients)`, `configure(layers, embd, heads, block, fresh)`,
  `load_weights(state, layers, embd, heads, block)`, `reset_params()`, `state_dict`, `param_keys`.
- **Tesztekben SOHA ne importáld a `name_ui`-t** — a modulimport cache-betöltést /
  háttér-tanítást indíthat (amely lefoglalja a `MODEL_LOCK`-ot, és lelassítja a
  teszteket). Tesztelj `microgpt` + `viz` szinten.
- A `model.pkl` cache formátuma: `{n_layer, n_embd, n_head, block_size, state_dict}`.
- A plotly.js-t a szerver **helyben** szolgálja ki (`/plotly.min.js`), CDN nélkül —
  ne cseréld vissza CDN-re.
- A `configure()`-nél `n_embd % n_head == 0` kötelező (különben ValueError).

## Ismert hibák / figyelmeztetések

- Régi (Value-alapú) `model.pkl` betöltése hibát dob → törölni kell.
- A `train()` (és a viz eldobható tréningjei) a `MODEL_LOCK` alatt futnak —
  tréning közben a `/plot/loss` és `/plot/nlayer` 409-et adnak (a galéria
  automatikusan újrapróbálja).
- Az `input.txt` hiánya esetén a `microgpt` import automatikusan letölti (internet).
- A viz `_snapshot`/`_restore` a TELJES konfigurációt menti/visszaállítja
  (n_layer, n_embd, n_head, block_size, shapes, param_keys, ...) — ha bővíted,
  mindkettőt frissítsd.
