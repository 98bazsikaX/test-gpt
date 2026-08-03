# microgpt

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

---

## Tartalomjegyzék

1. [Telepítés](#1-telepítés)
2. [Gyors indítás](#2-gyors-indítás)
3. [Fájlok és szerepük](#3-fájlok-és-szerepük)
4. [A létrejövő fájlok](#4-a-létrejövő-fájlok)
5. [Webes felület (name_ui.py)](#5-webes-felület-name_uipy)
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

## 1. Telepítés

**Előfeltételek**

- Python **3.12+** (`pyproject.toml` `requires-python = ">=3.12"`)
- [uv](https://docs.astral.sh/uv/) (gyors Python-csomagkezelő) — a venv-et és a
  függőségeket ez kezeli.

**Lépések**

```bash
# 1. A függőségek telepítése (létrehozza a .venv-et, és feloldja a uv.lock-ot)
uv sync

# 2. Ellenőrzés: a tesztek lefutnak
uv run pytest

# 3. Indítás (a böngésző automatikusan megnyílik)
uv run python name_ui.py
```

Ha hiányzik az `input.txt`, a `microgpt` modul **első importáláskor automatikusan
letölti** Karpathy names.txt-jét (internet szükséges).

---

## 2. Gyors indítás

| Mit akarsz? | Parancs |
|---|---|
| Böngészős felület (kiegészítés + tanítás + vizualizációk) | `uv run python name_ui.py` |
| Új nevek generálása (tréning + 20 minta) | `uv run python microgpt.py` |
| Egy előtag kiegészítése interaktívan | `uv run python complete_name.py` |
| Egy előtag kiegészítése argumentummal | `uv run python complete_name.py em` |
| Tesztek | `uv run pytest` |
| Lint / format ellenőrzés | `uv run ruff check .` |

---

## 3. Fájlok és szerepük

| Fájl | Szerep |
|---|---|
| `microgpt.py` | **A mag.** Adathalmaz betöltése, tokenizer, a GPT numpy-os előre- és visszaterjesztése, Adam, `train()`, `generate()`, `configure()`/`load_weights()`. Teljesen magyarul dokumentálva. |
| `name_ui.py` | **FastAPI webes felület.** Név-kiegészítés, temperature csúszka, újratanítás a böngészőből (háttér-szál + élő loss/PCA), `/viz` galéria, `model.pkl` cache-kezelés. |
| `viz.py` | **Plotly vizualizációk.** `loss_fig`, `nlayer_fig`, `pca_token_fig`, `pca_position_fig`, `attention_fig`, `distribution_fig`. Eldobható tréningeket futtat úgy, hogy a fő modellt nem bántja. |
| `complete_name.py` | **Parancssori név-kiegészítő.** Betanítja a modellt, majd egy előtagot kiegészít. |
| `.opencode/AGENTS.md` | **Ágens-memória** az opencode ágensei számára (projekt-összefoglaló, konvenciók, figyelmeztetések). |
| `.opencode/opencode.json` | Az opencode projekt-konfigja: a fenti `AGENTS.md`-t tölti be utasításként. |
| `tests/test_microgpt.py` | A mag tesztjei: forward/backward ekvivalencia az eredeti Value-autograd referenciával, tréning-determinizmus, `generate()` korlátai. |
| `tests/test_viz.py` | A konfiguráció (`configure`/`load_weights`), a `train()` callback és a vizualizáció-függvények tesztjei. |
| `conftest.py` | Üres pytest-gyökérfájl: ez teszi lehetővé, hogy a tesztek `import microgpt`-t használhassanak a projekt gyökeréből. |
| `pyproject.toml` | A uv-projekt leírása: függőségek, dev-függőségek, ruff és pytest beállítások. |
| `uv.lock` | A függőségek pontos, kipróbált verziói (ne szerkeszd kézzel). |
| `input.txt` | A tanítóadathalmaz (keresztnevek). **Nincs a repóban** — első futtatáskor automatikusan letöltődik a makemore-ból. |
| `model.pkl` | A betanított modell cache-e (létrejön, ld. [4. fejezet](#4-a-létrejövő-fájlok)). |

---

## 4. A létrejövő fájlok

| Fájl / mappa | Mikor jön létre? | Lehet törölni? |
|---|---|---|
| `input.txt` | Első `microgpt`-import, ha nincs meg (letölti). **Nincs a repóban** (gitignore). | Igen, de újra letölti. |
| `model.pkl` | A `name_ui.py` első indításakor, vagy ha `/retrain`-nel újratanítasz. A betanított súlyokat + konfigurációt tárolja (n_layer/n_embd/n_head/block_size + `state_dict`). | **Igen** — törlés után a következő indítás újratanít. |
| `.venv/` | `uv sync` / `uv add`. A virtuális környezet. | Igen, újrateremthető (`uv sync`). |
| `uv.lock` | `uv add` / `uv sync` | Nem kell törölni. |
| `__pycache__/`, `tests/__pycache__/` | Python importáláskor. | Igen, ártalmatlan. |
| `.pytest_cache/`, `.ruff_cache/` | `pytest` / `ruff` futtatáskor. | Igen, ártalmatlan. |
| `.python-version` | A `uv` esetenként létrehozza (a használt Python-verziót rögzíti). | Ha törlöd, a `uv` a `requires-python`-t használja. |

> **Fontos a cache-ről:** a `model.pkl` az *új* formátumban a konfigurációt is
> tárolja. Ha régi (csak súlyokat tartalmazó) cache van, a felület nem tudja
> hozzá a konfigurációt, ezért újratanít. Ha konfigurációt módosítasz, töröld a
> `model.pkl`-t, vagy tanítsd újra a felületről.

---

## 5. Webes felület (name_ui.py)

Indítás: `uv run python name_ui.py`

- A szerver **véletlenszerű (szabad) porton** indul — a konzol kiírja az URL-t
  (`open http://127.0.0.1:PORT ...`), és a böngésző automatikusan megnyílik.
- **Első indítás:** ha nincs `model.pkl`, a modell **háttérben** elkezd tanulni a
  jelenlegi (modul-szintű) konfigurációval. A felület azonnal elérhető, a
  tanulás folyamata élőben látható.
- **További indítások:** a `model.pkl` betöltése azonnali.

### Főoldal részei

1. **Név-kiegészítés** — írj be egy előtagot (pl. `ka`), és kapsz 5 kiegészítést.
2. **temperature csúszka** — 0.05..2 között; az érték élőben érvényesül a
   következő mintavételezésnél.
3. **Újratanítás** — állítható: `n_layer`, `n_embd`, `n_head`, `block_size`,
   `steps`, majd „Tanítás" gomb.
4. **Élő vizualizációk** a tanulás közben:
   - élő loss-görbe,
   - a 27 karakter-embedding élő PCA-ja (a pontok „mozognak”, ahogy a modell tanul).
5. **Link a galériához** (`/viz`).

---

## 6. API végpontok

| Módszer | Útvonal | Bemenet | Válasz |
|---|---|---|---|
| `GET` | `/` | — | A főoldal HTML-je. |
| `GET` | `/config` | — | Aktuális konfiguráció + temperature + fut-e tanítás. |
| `POST` | `/complete` | `{"seed": "ka"}` | `{"results": [...5 név...], "temperature": 0.5}` |
| `POST` | `/temperature` | `{"value": 1.3}` | `{"temperature": 1.3}` |
| `POST` | `/retrain` | `{"n_layer", "n_embd", "n_head", "block_size", "steps"}` | `{"ok": true}` (vagy 400/409 hibával) |
| `GET` | `/plot/live` | — | Élő tanítási állapot: `running`, `step`, `total`, `losses`, `final_loss`, `pca`. |
| `GET` | `/plot/loss` | — | A tréning loss-görbéje (Plotly JSON). |
| `GET` | `/plot/nlayer` | — | 1/2/4 rétegű modell loss-összehasonlítása. |
| `GET` | `/plot/pca_token` | — | Karakter-embeddingek PCA-ja. |
| `GET` | `/plot/pca_pos` | — | Pozíció-embeddingek PCA-ja. |
| `GET` | `/plot/attention` | `?name=karla` | Attention hőtérkép fejenként. |
| `GET` | `/plot/distribution` | `?prefix=ka` | A következő betű valószínűség-eloszlása. |
| `GET` | `/viz` | — | A vizualizáció-galería HTML-je. |
| `GET` | `/plotly.min.js` | — | A plotly.js, **helyben kiszolgálva** (internet/CDN nélkül). |

A Plotly-figurák JSON-ban érkeznek (`{"data": [...], "layout": {...}}`), amit a
böngészőben `Plotly.newPlot(el, data, layout)`-lel lehet megjeleníteni.

**Validációk:**

- `temperature`: 0.01..5
- `n_layer`: 1..8
- `n_embd`: 4..256
- `n_head`: 1..16, és `n_embd % n_head == 0` kell
- `block_size`: 2..64
- `steps`: 10..200 000

---

## 7. Parancssori használat

### Teljes tréning + 20 generált név

```bash
uv run python microgpt.py
```

A tréning a modul elején beállított `n_layer`, `n_embd`, `n_head`, `block_size`,
`num_steps` értékekkel fut (alapértelmezés szerint a `pyproject`-től független,
kódban rögzített értékek — a `microgpt.py` fejlécében módosíthatod).

### Név-kiegészítés

```bash
uv run python complete_name.py        # interaktívan kér előtagot
uv run python complete_name.py em     # előtag argumentumként
```

> A `complete_name.py` minden futáskor újratanít (nincs cache). Ha cache-elt
> modellt akarsz, használd a webes felületet.

---

## 8. Hogyan működik a modell

Ez a mikroGPT egy egyrétegű (alapesetben), karakter-szintű GPT. A teljes
folyamat magyarul dokumentálva van a `microgpt.py`-ban; itt csak a lényeg:

1. **Tokenizer** — minden egyedi karakter token id-t kap (0..n-1), plusz egy
   `BOS` token jelöli a szekvencia elejét/végét.
2. **Beágyazás** — `token_embedding` + `position_embedding` minden tokenre.
3. **Transformer-blokk** — RMSNorm, kauzális multi-head self-attention, MLP
   (ReLU), két residual összekötéssel.
4. **lm_head** — a rejtett rétegből a szókészletre vetít, logiteket adva, hogy
   mi következhet a szekvenciában.
5. **Tréning** — batched forward + kézzel levezetett backward (láncszabály)
   numpy `float32` tömbökkel, Adam optimalizálóval, lineáris lr-csökkenéssel.
6. **Inferencia** — `generate()` tokenenként mintavételez a logitek
   softmaxából, `temperature`-rel szabályozhatóan.

Az eredeti Value-alapú autograd helyett **vektorizált, kézi backward** dolgozik,
ezért a tréning kb. 50-100x gyorsabb — a matematika viszont pontosan ugyanaz
(ext a tesztek a referencia-autograddal ellenőrzik).

---

## 9. Hiperparaméterek útmutató

| Paraméter | Jelentés | Tipp |
|---|---|---|
| `n_layer` | A transformer mélysége (rétegszám). | Több réteg = több elvont minta (szótagszerkezet, suffixek), de több lépés kell. |
| `n_embd` | A reprezentáció szélessége. | 16 a minimál-játék; 32-64 már érezhetően „ügyesebb”, de lassabb és több paraméter. |
| `n_head` | Az attention fejek száma. | `n_embd`-nek oszthatónak kell lennie vele. |
| `block_size` | A kontextusablak hossza. | A leghosszabb név 15 karakter, 16 elég. |
| `steps` | Hány dokumentumot lát a modell a tréning alatt. | A loss a lépésszámmal csökken; kezdésnek 1000-5000, minőségért több. |
| `temperature` | A mintavételezés „kreativitása”. | 0.1 = nagyon konzervatív, 0.5 = jó egyensúly, 1+ = kaotikusabb. |

Ha `n_layer`-t növeled, **növeld a `steps`-et is** — különben a mélyebb modell
alultanul. A UI `/plot/nlayer` galériája mutatja a 1/2/4 réteg közötti
különbséget azonos lépésszámon.

---

## 10. Vizualizációk

A `/viz` galériában (és részben a főoldalon) interaktív, Plotly-alapú ábrák:

| Ábra | Mit mutat? |
|---|---|
| **Élő loss** | A háttér-tanítás loss-görbéje, frissül közben. |
| **Élő embedding-PCA** | A 27 karakter-embedding 2D-s vetülete; a tanulás alatt látszik, ahogy csoportosulnak. |
| **Rétegszám-összehasonlítás** | 1/2/4 réteg loss-görbéi azonos lépésszámon. |
| **Token PCA** | A karakterek token-embeddingjének PCA-ja (vajon csoportosulnak a magánhangzók?). |
| **Pozíció PCA** | A pozíció-embeddingek PCA-ja (gyakran „hullámos” mintát ad). |
| **Attention hőtérkép** | Egy névre, fejenként: melyik betű melyik korábbira figyel (kauzális maszk). |
| **Eloszlás** | Egy prefix után: mennyire valószínű az egyes következő betűk. |

> A `/plot/*` ábrák Plotly-JSON-t adnak. A plotly.js-t a szerver **helyben**
> szolgálja ki (`/plotly.min.js`), így a megjelenítéshez nincs szükség internetre.
> Ha épp tanítás fut, a `/plot/loss` és `/plot/nlayer` 409-es választ adnak —
> a galéria ilyenkor automatikusan újrapróbálja (2 mp-enként, amíg a tanítás
> nem fejeződik be).

---

## 11. Tesztek és kódminőség

```bash
uv run pytest         # 15 teszt
uv run ruff check .   # lint
```

A tesztek közül a legérdekesebb a `tests/test_microgpt.py`-ben található
**referencia-gradiens teszt**: szó szerint lefuttatja az eredeti Value-alapú
microgpt-et (a karpathy gist alapján), és összehasonlítja a numpy-os backward
által számolt gradienseket a referencia-autogradéval. Ezzel garantáljuk, hogy
a vektorizált átírat **ugyanazt** tanulja, mint az eredeti.

---

## 12. Teljesítmény

A numpy-os átírat kb. **50-100x gyorsabb**, mint a tiszta-Python Value-verzió:

| Konfiguráció | Idő (1000 lépés) |
|---|---|
| 1 réteg, 16 embd | ~0.5 s |
| 4 réteg, 16 embd | ~2 s |
| 4 réteg, 32 embd | ~8-10 s |

A `train()` a végén kiírja a pontos időzítést (`trained N steps in X.XXs`).

---

## 13. Hibakeresés és gyakori kérdések

**„Can't get attribute 'Value'” hiba a `model.pkl` betöltésekor**
Régi, a Value-alapú verzióból származó cache van jelen. Töröld a `model.pkl`-t.

**„n_embd osztható legyen n_head-dal” hiba**
A `n_embd` és `n_head` nem oszthatók egymással (pl. 16 és 3). Válassz párosítást,
ahol `n_embd % n_head == 0`.

**A galéria ábrái nem jelennek meg**
A plotly.js-t a szerver helyben adja ki (`/plotly.min.js`), internet nem kell.
Ha mégis üres egy ábra: (1) a tréning alatt a `/plot/loss` és `/plot/nlayer`
409-et adnak — a galéria automatikusan újrapróbálja, várj egy kicsit;
(2) nyisd meg a böngésző konzolt (F12), és nézd meg, van-e hibaüzenet;
(3) ellenőrizd, hogy a szerver fut és a `/plotly.min.js` elérhető.

**A modell értelmetlen neveket ad**
A tréning túl rövid, vagy a `temperature` túl magas. Taníts tovább (`steps`),
vagy csökkentsd a temperature-t.

**Melyik porton fut a szerver?**
Véletlenszerű (szabad) porton — az indításkor kiírt `open http://127.0.0.1:PORT`
URL-t nyisd meg (a böngésző magától megnyílik).

**Hogyan törölhetem a tanult modellt és kezdem újra?**
Töröld a `model.pkl`-t (és opcionálisan az `input.txt`-et), majd indítsd újra.

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
