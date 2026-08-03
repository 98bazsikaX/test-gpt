# microgpt

![CI](https://github.com/98bazsikaX/test-gpt/actions/workflows/ci.yml/badge.svg)

Egy **GPT a legapróbb, oktató jellegű formájában** — tiszta Python + numpy, sémény
keretrendszer nélkül. A modell azt tanulja meg, hogyan néznek ki a keresztnevek,
majd képes új neveket generálni, illetve megadott név-előtagokat kiegészíteni.

Az eredeti ötlet Karpathy [microgpt](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py) fájlján alapul. Ez a projekt
**numpy-ra vektorizált átírat** (~50-100x gyorsabb tréning, azonos matematika),
ráadásul kap hozzá:

- böngészős felületet (FastAPI) név-kiegészítéssel,
- **újratanítást a böngészőből** állítható hiperparaméterekkel,
- **élő** vizualizációkat (loss-görbe, karakter-embeddingek PCA-ja a tanulás közben),
- vizualizáció-galeriát (attention hőtérképek, eloszlások, rétegszám-összehasonlítás).

A projekt a `my-little-jpa` mintájára **monorepo**: `backend/` (Python/FastAPI) +
`frontend/` (statikus web, nginx), docker-compose-zal indítható.

---

## Tartalomjegyzék

1. [Repó-szerkezet](#1-repó-szerkezet)
2. [Telepítés](#2-telepítés)
3. [Gyors indítás](#3-gyors-indítás)
4. [Docker](#4-docker)
5. [Webes felület](#5-webes-felület)
6. [API végpontok](#6-api-végpontok)
7. [Parancssori használat](#7-parancssori-használat)
8. [Hogyan működik a modell](#8-hogyan-működik-a-modell)
9. [Hiperparaméterek útmutató](#9-hiperparaméterek-útmutató)
10. [Vizualizációk](#10-vizualizációk)
11. [Tesztek és kódminőség](#11-tesztek-és-kódminőség)
12. [Teljesítmény](#12-teljesítmény)
13. [Hibakeresés és gyakori kérdések](#13-hibakeresés-és-gyakori-kérdések)
14. [Licenc és hivatkozások](#14-licenc-és-hivatkozások)

---

## 1. Repó-szerkezet

```
microgpt/
├── backend/                  # Python: a modell + FastAPI + vizualizációk
│   ├── microgpt.py           # a mag (tokenizer, forward/backward, train, generate)
│   ├── name_ui.py            # a FastAPI "backend" (API + statikus frontend kiszolgálás)
│   ├── viz.py                # Plotly vizualizációk
│   ├── complete_name.py      # parancssori név-kiegészítő
│   ├── server.py             # a Docker-konténer indítója (uvicorn, 8000)
│   ├── pyproject.toml        # uv-projekt (függőségek, ruff, pytest)
│   ├── uv.lock
│   ├── conftest.py           # pytest-gyökér
│   └── tests/                # pytest tesztek
├── frontend/                 # a böngészős rész (statikus, nginx szolgálja ki)
│   ├── index.html            # főoldal (kiegészítés + tanítás + élő ábrák)
│   ├── viz.html              # vizualizáció-galería
│   ├── app.js                # a főoldal JS-e
│   ├── viz.js                # a galéria JS-e
│   ├── style.css
│   ├── nginx.conf            # statikus + API-proxy a backendre
│   └── Dockerfile
├── docker-compose.yml        # backend (uvicorn) + frontend (nginx)
├── PLAN.md                   # fejlesztési terv / mérföldkövek
├── README.md
├── LICENSE                   # MIT
├── .editorconfig
├── .gitignore
└── .opencode/                # opencode projekt-config
    ├── opencode.json
    └── AGENTS.md             # ágens-memória (a projektről, konvenciókról)
```

| Fájl | Szerep |
|---|---|
| `backend/microgpt.py` | **A mag.** Adathalmaz betöltése, tokenizer, numpy-os előre- és visszaterjesztés, Adam, `train()`, `generate()`, `configure()`/`load_weights()`. |
| `backend/name_ui.py` | **FastAPI backend.** Név-kiegészítés, temperature, újratanítás (háttér-szál), élő loss/PCA, galéria-API, `model.pkl` cache. |
| `backend/viz.py` | **Plotly vizualizációk.** `loss_fig`, `nlayer_fig`, `pca_token_fig`, `pca_position_fig`, `attention_fig`, `distribution_fig`. |
| `backend/complete_name.py` | Parancssori név-kiegészítő (minden futáskor újratanít). |
| `backend/server.py` | Docker-indító: `_load_cache()` + `uvicorn` 8000-es porton. |
| `frontend/*` | A böngészős felület (HTML/JS/CSS). Helyben a backend szolgálja ki, docker-ben az nginx. |
| `docker-compose.yml` | Két szolgáltatás: `backend` (uvicorn) + `frontend` (nginx). |
| `.opencode/AGENTS.md` | Ágens-memória az opencode számára. |

---

## 2. Telepítés

**Előfeltételek**

- Python **3.12+** (`pyproject.toml` `requires-python = ">=3.12"`)
- [uv](https://docs.astral.sh/uv/) — a venv-et és a függőségeket ez kezeli.
- (Dockerhez: Docker Desktop vagy más docker engine.)

**Lépések**

```bash
# 1. A függőségek telepítése (backend/.venv)
cd backend
uv sync

# 2. Tesztek
uv run pytest

# 3. Indítás (a böngésző automatikusan megnyílik)
uv run python name_ui.py
```

Ha hiányzik a `backend/input.txt`, a `microgpt` modul **első importáláskor
automatikusan letölti** Karpathy names.txt-jét (internet szükséges).

---

## 3. Gyors indítás

| Mit akarsz? | Parancs |
|---|---|
| Böngészős felület | `cd backend && uv run python name_ui.py` |
| Új nevek generálása (tréning + 20 minta) | `cd backend && uv run python microgpt.py` |
| Egy előtag kiegészítése interaktívan | `cd backend && uv run python complete_name.py` |
| Egy előtag kiegészítése argumentummal | `cd backend && uv run python complete_name.py em` |
| Tesztek | `cd backend && uv run pytest` |
| Lint / format ellenőrzés | `cd backend && uv run ruff check .` |

---

## 4. Docker

```bash
docker compose up --build
```

- **Frontend:** http://localhost:8080 (nginx, a statikus fájlok)
- **Backend:** http://localhost:8000 (uvicorn, az API és a plotly.js)

A frontend-konténer az API-kéréseket (`/complete`, `/config`, `/retrain`,
`/temperature`, `/plot/*`, `/plotly.min.js`) proxy-zza a backendre.

> Megjegyzés: a konténerek alapból **állapotmentesek** — az `input.txt` és a
> `model.pkl` a konténer újraindításakor újra létrejön (első induláskor a modell
> letölti az adatot, majd háttérben tanul). Perzisztens volumen hozzáadása a
> `PLAN.md`-ben szerepel.

---

## 5. Webes felület

Indítás: `cd backend && uv run python name_ui.py`

- A szerver **véletlenszerű (szabad) porton** indul — a konzol kiírja az URL-t,
  és a böngésző automatikusan megnyílik.
- **Első indítás:** ha nincs `model.pkl`, a modell **háttérben** elkezd tanulni a
  jelenlegi konfigurációval. A felület azonnal elérhető, a tanulás élőben látható.

### Főoldal részei

1. **Név-kiegészítés** — előtag beírása (pl. `ka`), 5 kiegészítés.
2. **temperature csúszka** — 0.05..2 között, élőben érvényesül.
3. **Újratanítás** — `n_layer`, `n_embd`, `n_head`, `block_size`, `steps`.
4. **Élő vizualizációk** — loss-görbe + karakter-embeddingek PCA-ja a tanulás közben.
5. **Link a galériához** (`/viz`).

---

## 6. API végpontok

| Módszer | Útvonal | Bemenet | Válasz |
|---|---|---|---|
| `GET` | `/` | — | A főoldal (frontend/index.html). |
| `GET` | `/viz` | — | A galéria (frontend/viz.html). |
| `GET` | `/app.js`, `/viz.js`, `/style.css` | — | A frontend statikus fájljai. |
| `GET` | `/config` | — | Aktuális konfiguráció + temperature + fut-e tanítás. |
| `POST` | `/complete` | `{"seed": "ka"}` | `{"results": [...5 név...], "temperature": 0.5}` |
| `POST` | `/temperature` | `{"value": 1.3}` | `{"temperature": 1.3}` |
| `POST` | `/retrain` | `{"n_layer", "n_embd", "n_head", "block_size", "steps"}` | `{"ok": true}` (vagy 400/409 hibával) |
| `GET` | `/plot/live` | — | Élő tanítási állapot: `running`, `step`, `total`, `losses`, `pca`. |
| `GET` | `/plot/loss` | — | A tréning loss-görbéje (Plotly JSON). |
| `GET` | `/plot/nlayer` | — | 1/2/4 rétegű modell loss-összehasonlítása. |
| `GET` | `/plot/pca_token` | — | Karakter-embeddingek PCA-ja. |
| `GET` | `/plot/pca_pos` | — | Pozíció-embeddingek PCA-ja. |
| `GET` | `/plot/attention` | `?name=karla` | Attention hőtérkép fejenként. |
| `GET` | `/plot/distribution` | `?prefix=ka` | A következő betű valószínűség-eloszlása. |
| `GET` | `/plotly.min.js` | — | A plotly.js, helyben (internet nélkül). |

A Plotly-figurák JSON-ban érkeznek (`{"data": [...], "layout": {...}}`).

**Validációk:** `temperature` 0.01..5; `n_layer` 1..8; `n_embd` 4..256;
`n_head` 1..16 (`n_embd % n_head == 0`); `block_size` 2..64; `steps` 10..200 000.

---

## 7. Parancssori használat

```bash
cd backend
uv run python microgpt.py                 # tréning + 20 generált név
uv run python complete_name.py            # interaktív előtag
uv run python complete_name.py em         # előtag argumentummal
```

---

## 8. Hogyan működik a modell

Ez a mikroGPT egy karakter-szintű GPT. A teljes folyamat magyarul dokumentálva
van a `backend/microgpt.py`-ban; a lényeg:

1. **Tokenizer** — minden egyedi karakter token id-t kap, plusz egy `BOS` token.
2. **Beágyazás** — `token_embedding` + `position_embedding`.
3. **Transformer-blokk** — RMSNorm, kauzális multi-head self-attention, MLP (ReLU),
   két residual összekötéssel.
4. **lm_head** — logitek arra, hogy mi következhet a szekvenciában.
5. **Tréning** — batched forward + kézzel levezetett backward numpy `float32`
   tömbökkel, Adam optimalizálóval, lineáris lr-csökkenéssel.
6. **Inferencia** — `generate()` tokenenként mintavételez a logitek softmaxából,
   `temperature`-rel szabályozhatóan.

Az eredeti Value-alapú autograd helyett **vektorizált, kézi backward** dolgozik,
ezért a tréning kb. 50-100x gyorsabb — a matematika viszont pontosan ugyanaz
(ext a tesztek a referencia-autograddal ellenőrzik).

---

## 9. Hiperparaméterek útmutató

| Paraméter | Jelentés | Tipp |
|---|---|---|
| `n_layer` | Mélység. | Több réteg = elvontabb minták, de több lépés kell. |
| `n_embd` | Szélesség. | 16 a minimál; 32-64 ügyesebb, de lassabb. |
| `n_head` | Fejek száma. | `n_embd % n_head == 0` kell. |
| `block_size` | Kontextusablak. | A leghosszabb név 15 karakter, 16 elég. |
| `steps` | Tanulási lépések. | Loss a lépésszámmal csökken; kezdésnek 1000-5000. |
| `temperature` | „Kreativitás". | 0.1 konzervatív, 0.5 jó egyensúly, 1+ kaotikusabb. |

---

## 10. Vizualizációk

A `/viz` galériában (és részben a főoldalon) interaktív, Plotly-alapú ábrák:
élő loss, élő embedding-PCA, rétegszám-összehasonlítás, token/pozíció PCA,
attention hőtérképek, következő-betű eloszlás.

A `/plot/*` ábrák Plotly-JSON-t adnak; a plotly.js-t a szerver **helyben**
szolgálja ki (`/plotly.min.js`), így internet nem kell. Ha épp tanítás fut,
a `/plot/loss` és `/plot/nlayer` 409-et adnak — a galéria automatikusan
újrapróbálja (2 mp-enként), amíg a tanítás nem fejeződik be.

---

## 11. Tesztek és kódminőség

```bash
cd backend
uv run pytest         # 15 teszt
uv run ruff check .   # lint
```

A legérdekesebb a `tests/test_microgpt.py` **referencia-gradiens tesztje**: az
eredeti Value-alapú microgpt-et (a karpathy gist alapján) futtatja, és a
numpy-os backward gradienseit hasonlítja a referencia-autogradéval.

---

## 12. Teljesítmény

A numpy-os átírat kb. **50-100x gyorsabb**, mint a tiszta-Python Value-verzió:

| Konfiguráció | Idő (1000 lépés) |
|---|---|
| 1 réteg, 16 embd | ~0.5 s |
| 4 réteg, 16 embd | ~2 s |
| 4 réteg, 32 embd | ~8-10 s |

---

## 13. Hibakeresés és gyakori kérdések

**„Can't get attribute 'Value'" a `model.pkl` betöltésekor**
Régi, Value-alapú cache van jelen. Töröld a `backend/model.pkl`-t.

**„n_embd osztható legyen n_head-dal"**
A két érték nem osztható egymással. Válassz párosítást, ahol `n_embd % n_head == 0`.

**A galéria ábrái nem jelennek meg**
A plotly.js-t a szerver helyben adja ki (`/plotly.min.js`), internet nem kell.
Ha mégis üres: (1) tréning alatt a `/plot/loss` és `/plot/nlayer` 409-et adnak —
a galéria automatikusan újrapróbálja; (2) böngészőkonzol (F12); (3) ellenőrizd,
hogy a backend elérhető.

**Melyik porton fut a szerver?**
Helyben véletlenszerű (a konzol kiírja); dockerben frontend 8080, backend 8000.

**A modell értelmetlen neveket ad**
Túl rövid tréning, vagy túl magas `temperature`. Taníts tovább, vagy csökkentsd.

**Hogyan kezdem újra?**
Töröld a `backend/model.pkl`-t (és opcionálisan a `backend/input.txt`-et).

---

## 14. Licenc és hivatkozások

- **Ez a repo** MIT licencű — ld. a `LICENSE` fájlt (Copyright © 2026 Balázs Szombati).
- Az algoritmus Andrej Karpathy
  [microgpt](https://gist.githubusercontent.com/karpathy/8627fe009c40f57531cb18360106ce95/raw/14fb038816c7aae0bb9342c2dbf1a51dd134a5ff/microgpt.py)
  gist-jéből származik (a gist-en nincs explicit licenc; a `@karpathy` feliratot
  a kód és ez a README is megtartja). Ez a projekt ennek a vektorizált, numpy-os
  átirata, valamint felületi és vizualizációs bővítése — tanulási céllal.
- A `names.txt` adathalmaz a [karpathy/makemore](https://github.com/karpathy/makemore)
  repóból származik, amely **MIT licencű** (Copyright 2022 Andrej Karpathy) —
  az adat az Egyesült Államok Társadalombiztosítási Hivatala (SSA) nyilvános
  babanev-adatain alapul.
