"""
=============================================================================
 microgpt — a GPT legapróbb, tiszta Python + numpy implementációja
=============================================================================

Ez a fájl egy teljes, minimál-GPT *tréning* és *inferencia* algoritmusa.
Minden "mágia" itt van a kódban: nincs semmilyen gépi tanulási keretrendszer,
még autograd könyvtár sem. A szükséges matematikát (visszaterjesztés, azaz a
láncszabály) kézzel, numpy tömbökkel írtuk meg.

Miért numpy, és nem a korábbi tiszta-Python `Value` autograd?
    Az eredeti mikroGPT minden skalaros számítást `Value` objektumokon
    végzett, és egy hatalmas számítási gráfot épített a visszaterjesztéshez.
    Ez rendkívül lassú volt. Itt a vektorizált numpy `float32` tömbök
    "egyszerre" számolják ki a teljes mátrixműveleteket, a gradienseket
    pedig a láncszabályt követve, op-ról-opra vezetjük le kézzel.
    Eredmény: ~10-50x gyorsabb tréning, pontosan ugyanaz a modell.

A fájl felépítése (olvasd sorban!):
    1) Adathalmaz betöltése
    2) Tokenizer (karakter -> token id)
    3) A modell paramétereinek inicializálása
    4) Alap műveletek: softmax, RMSNorm (+ gradienseik)
    5) Előrehaladás (forward) — az egész szekvenciát egyszerre
    6) Visszaterjesztés (backward) — kézi láncszabály
    7) Adam optimalizáló + tréning ciklus
    8) Inferencia: `generate()` — név-generálás, megadott előtaggal

Minden függvény docstring-je magyarázza a *miért*-et és a képleteket is.
"""

# -----------------------------------------------------------------------------
# 0) Importok és a véletlenszám-generátorok "beidomítása"
# -----------------------------------------------------------------------------
import os  # fájllétezés ellenőrzése (`input.txt`)
import random  # a dokumentumok összekeverése (shuffle)
import threading  # a tréningek összehangolása (MODEL_LOCK)
import time  # a tréning időzítése (benchmark)

import numpy as _np  # a vektorizált számolások alapja (CPU)

# Opcionális GPU-backend (CuPy): ha elérhető, a modell GPU-n tanul, egyébként
# (pl. CI-n) automatikusan numpy-fallback. A cupy "drop-in" numpy, így a
# kézi forward/backward kód változatlan marad.
try:
    import cupy as _gpu
    _GPU_AVAILABLE = bool(_gpu.cuda.is_available())
except Exception:  # nincs cupy vagy nincs CUDA-driver
    _gpu = None
    _GPU_AVAILABLE = False

np = _gpu if _GPU_AVAILABLE else _np
BACKEND = 'gpu' if _GPU_AVAILABLE else 'cpu'


def set_backend(name):
    """A számolási backend átváltása: 'cpu' (numpy) vagy 'gpu' (cupy).

    Megjegyzés: a már meglévő paraméterek (state_dict) nem "költöznek" át —
    váltás után hívd a reset_params()-t, ha a kiválasztott backendre szóló
    friss paramétereket akarsz.
    """
    global np, BACKEND
    if name == 'gpu':
        if not _GPU_AVAILABLE:
            raise RuntimeError('cupy/CUDA nem elérhető')
        np = _gpu
        BACKEND = 'gpu'
    elif name == 'cpu':
        np = _np
        BACKEND = 'cpu'
    else:
        raise ValueError(f'ismeretlen backend: {name!r} (cpu|gpu)')


def _to_cpu(x):
    """Egy tömböt CPU-numpy-ra konvertál (cupy esetén `.get()`)."""
    return x.get() if hasattr(x, 'get') else x


def _to_active(x):
    """Egy (CPU-s) tömböt az aktív backendre konvertál (súlybetöltéshez)."""
    return np.asarray(x)


random.seed(42)    # Legyen rend a káoszban (reprodukálható shuffle)
np.random.seed(42) # és a backend véletlenjei is legyenek reprodukálhatók

# Egyetlen zár a modell-gróbokra: a tréningek (UI-háttérszál, viz-eldobható
# tréning) egymást kizárják. Az inferencia (completions, plotok) nem zárol,
# hogy a webes felület élőben követhessen egy futó tanítást.
MODEL_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# 1) Adathalmaz: `docs` — dokumentumok (keresztnevek) listája
# -----------------------------------------------------------------------------
# Ha nincs meg az `input.txt`, letöltjük Karpathy names.txt-jét.
# Minden sor egy-egy keresztnév (kisbetűvel). Ezekből tanulja meg a modell,
# milyen "betű-sorozatok" adnak ki valószerű nevet.
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')

docs = [line.strip() for line in open('input.txt') if line.strip()]
random.shuffle(docs)  # a tréning sorrendje véletlenszerű legyen
print(f"num docs: {len(docs)}")


# -----------------------------------------------------------------------------
# 2) Tokenizer: karakterek <-> egész számok ("tokenek")
# -----------------------------------------------------------------------------
# A legegyszerűbb tokenizer: minden egyedi karakter kap egy token id-t (0..n-1).
# Ezen kívül bevezetünk egy speciális BOS ("Begin Of Sequence") tokent is,
# amely a szöveg elejét jelöli. A BOS token id-je a karakterek száma, tehát a
# teljes szókészlet (vocab) mérete = karakterek száma + 1.
uchars = sorted(set(''.join(docs)))       # az egyedi karakterek rendezve
stoi = {ch: i for i, ch in enumerate(uchars)}  # karakter -> id ("string to int")
BOS = len(uchars)                          # a BOS token id-ja
vocab_size = len(uchars) + 1               # teljes szókészlet mérete
print(f"vocab size: {vocab_size}")


# -----------------------------------------------------------------------------
# 3) A modell konfigurációja és paramétereinek inicializálása
# -----------------------------------------------------------------------------
# Architektúra (GPT-2 stílusú, leegyszerűsítve):
#   - n_layer réteg (depth), mindegyikben: multi-head attention + MLP blokk
#   - n_embd dimenziós reprezentáció (width)
#   - block_size token hosszú kontextusablak (a leghosszabb név 15 karakter)
#   - n_head attention fej; minden fej head_dim dimenziót lát
# Megjegyzés: az eredeti GPT-2-höz képest kimarad a LayerNorm (helyette
# RMSNorm), nincsenek bias-ok, és GeLU helyett ReLU aktivációt használunk.
n_layer = 6     # a transformer mélysége (rétegszám)
n_embd = 16     # a háló szélessége (beágyazási dimenzió)
block_size = 16 # a kontextusablak maximális hossza (tokenekben)
n_head = 4      # az attention fejek száma
head_dim = n_embd // n_head  # egy-egy fej dimenziója (16 / 4 = 4)


def init(shape, std=0.08):
    """Egy paramétertömb inicializálása véletlenszerűen, Gauss-eloszlással.

    A kis szórás (0.08) azért jó, mert a tréning elején a logitek se legyenek
    túl nagyok (különben a softmax tele lenne 0/1 közeliekre eső értékekkel).
    `float32` lesz, mert az elég a tanuláshoz és gyorsabb, mint a float64.
    """
    return (np.random.randn(*shape) * std).astype(np.float32)


def build_param_keys(layers):
    """Összeállítja a paraméterek kanonikus sorrendjét egy adott mélységhez.

    Fontos, hogy a sorrend mindig ugyanaz legyen, mert a model.pkl cache-be
    mentéskor és visszatöltéskor ehhez igazodunk (és az Adam pufferek is ehhez
    a sorrendhez tartoznak).
    Érthetőbb nevek az eredeti wte/wpe/attn_wq... helyett:
      token_embedding   : minden token id-hez egy n_embd dimenziós vektor
      position_embedding: minden pozícióhoz egy n_embd dimenziós vektor
      lm_head           : az utolsó rejtett rétegből a szókészletre (logitek)
      query_w/key_w/value_w/output_w : az attention mátrixai
      fc1_w/fc2_w       : az MLP két mátrixa (expand / összenyomás)
    """
    keys = ['token_embedding', 'position_embedding', 'lm_head']
    for i in range(layers):
        keys += [f'layer{i}.query_w', f'layer{i}.key_w', f'layer{i}.value_w',
                 f'layer{i}.output_w', f'layer{i}.fc1_w', f'layer{i}.fc2_w']
    return keys


def build_shapes():
    """Az egyes paraméterek alakjai az aktuális konfigurációhoz.

    Az `n_embd`/`block_size` változásával ezt újra kell építeni (a
    `configure()` megteszi), ezért függvény, nem fix modul-szintű dict.
    """
    return {
        'token_embedding': (vocab_size, n_embd),
        'position_embedding': (block_size, n_embd),
        'lm_head': (vocab_size, n_embd),
        'query_w': (n_embd, n_embd), 'key_w': (n_embd, n_embd),
        'value_w': (n_embd, n_embd), 'output_w': (n_embd, n_embd),
        'fc1_w': (4 * n_embd, n_embd),   # a rejtett réteg 4x szélesebb (GPT-2 stílus)
        'fc2_w': (n_embd, 4 * n_embd),
    }


param_keys = build_param_keys(n_layer)
shapes = build_shapes()


def reset_params():
    """A paraméterek újrainicializálása (pl. tesztekhez, friss indításhoz)."""
    global state_dict, params
    state_dict = {}
    for key in param_keys:
        if '.' in key:  # réteg-specifikus paraméter (pl. layer0.query_w)
            layer_key = key.split('.')[1]
            state_dict[key] = init(shapes[layer_key])
        else:           # globális paraméter (token/pozíció/lm_head)
            state_dict[key] = init(shapes[key])
    params = [state_dict[key] for key in param_keys]


def configure(layers=None, embd=None, heads=None, block=None, fresh=True):
    """Beállítja a modell méreteit, és (opcionálisan) új paramétereket ad.

    Bármelyik argumentum elhagyható — akkor az aktuális érték marad.
    A `fresh=True` új, véletlenszerű paramétereket inicializál; a `fresh=False`
    csak a konfigurációt állítja át (a súlyok betöltéséhez, ld. load_weights).

    Megkötések:
      - n_embd osztható kell legyen n_head-dal (minden fej egyenlő méretű)
      - block_size >= 2 (legalább "bemenet + következő token" kell)
    """
    global n_layer, n_embd, n_head, block_size, head_dim, param_keys, shapes
    if layers is not None:
        n_layer = layers
    if embd is not None:
        n_embd = embd
    if heads is not None:
        n_head = heads
    if block is not None:
        block_size = block
    if n_embd % n_head != 0:
        raise ValueError(f'n_embd ({n_embd}) osztható kell legyen n_head-dal ({n_head})')
    if block_size < 2:
        raise ValueError(f'block_size legalább 2 legyen (most {block_size})')
    head_dim = n_embd // n_head
    param_keys = build_param_keys(n_layer)
    shapes = build_shapes()
    if fresh:
        reset_params()


def set_config(layers):
    """Átállítja a rétegszámot, és friss paramétereket inicializál.

    Ez az "eldobható" (throwaway) tréningekhez kell: pl. a vizualizáció
    összehasonlítja a 1/2/4 rétegű modelleket. A hívó feladata, hogy utána
    visszaállítsa az eredeti n_layer/param_keys/state_dict/params értékeket
    (a viz modul ezt megteszi).
    """
    configure(layers=layers, fresh=True)


def load_weights(state, layers, embd=None, heads=None, block=None):
    """Betölt egy elmentett súlykészletet a megadott konfigurációhoz.

    A `model.pkl` cache-nek a rétegszámot és a méreteket is tárolnia kell,
    különben újraindításkor pl. a 4 rétegű súlyok egy 1 rétegű modellbe
    kerülnének. A betöltött (CPU-s) súlyokat az aktív backendre konvertáljuk,
    így cache-betöltés után is GPU-n tanulhatunk.
    """
    global state_dict, params
    configure(layers=layers, embd=embd, heads=heads, block=block, fresh=False)
    state_dict = {key: _to_active(arr) for key, arr in state.items()}
    params = [state_dict[key] for key in param_keys]


reset_params()
num_params = sum(int(p.size) for p in params)
print(f"num params: {num_params}")


# -----------------------------------------------------------------------------
# 4) Alap műveletek és a visszaterjesztéshez szükséges gradienseik
# -----------------------------------------------------------------------------

def softmax(values, axis=-1):
    """Numerikusan stabil softmax.

    Softmax(logit_i) = exp(logit_i - max) / sum_j exp(logit_j - max).
    A maximum kivonása csak numerikus okokból történik (ne legyen túlcsordulás),
    az eredményt matematikailag nem változtatja meg.
    """
    values = values - values.max(axis=axis, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum(axis=axis, keepdims=True)


def rmsnorm(x):
    """RMS (Root Mean Square) normalizálás a dimenziók mentén.

    y = x / sqrt( mean(x^2) + eps )

    A LayerNorm-tól eltérően nincs benne eltolás és skála paraméter; elég,
    ha a vektor hosszát egységnyire állítjuk, hogy a számok stabilak maradjanak.
    Az eps (1e-5) a nullával való osztástól véd.
    """
    mean_square = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(mean_square + 1e-5)


def _rmsnorm_backward(output, x, grad_output):
    """Az RMSNorm visszaterjesztése.

    Legyen  s = mean(x^2) + eps  és  scale = s^-0.5,  ekkor  y = x * scale.
    A láncszabály szerint (levezetve: dy_i/dx_j = scale*δ_ij - scale^3 * x_i*x_j / d):

        dx = grad_output * scale
             - scale^3 * x * mean(x * grad_output)     (átlag a dimenziókra)

    Vagyis a "saját" gradiense (első tag) mellé kapunk egy korrekciós tagot,
    ami a vektor *hossza* megváltozásából származik.
    """
    mean_square = (x * x).mean(axis=-1, keepdims=True) + 1e-5
    scale = mean_square ** -0.5
    return grad_output * scale - (scale ** 3) * x * (grad_output * x).mean(axis=-1, keepdims=True)


# -----------------------------------------------------------------------------
# 5) Előrehaladás (forward): az egész szekvencia, egyetlen menetben
# -----------------------------------------------------------------------------
# Az eredeti mikroGPT pozíciónként, egyesével vezette végig a tokent, és a
# `keys`/`values` listákba gyűjtötte a múltat, hogy ne kelljen maszk.
# Itt ezt felváltjuk a szabványos GPT tréninggel: az egész szekvenciát egyszerre
# futtatjuk, és egy kauzális (háromszög) maszkkal "megtiltjuk" a modellnek,
# hogy a jövőbe nézzen. Matematikailag ez ugyanazt adja, csak gyorsabb.

def forward(tokens):
    """Előrehaladás a teljes token-szekvencián.

    Bemenet: `tokens` — int tömb, hossza T <= block_size.
    Kimenet: `(logits, cache)`:
        - logits: (T, vocab_size) — minden pozícióra, mi jöhet legközelebb
        - cache : minden, amire a backward-nak szüksége lesz (köztes értékek)

    A forward lépései (rétegenként):
      1. Beágyazás: x = token_embedding[tokens] + position_embedding[:T]
      2. Első RMSNorm (a residual-ok miatt nem redundáns!)
      3. Blokk:
         a. RMSNorm -> query/key/value lekérdezése
         b. kauzális multi-head attention: scores = QK^T/sqrt(d), softmax, sum V
         c. output vetítés + residual összekötés
         d. MLP: RMSNorm -> fc1 -> ReLU -> fc2 -> residual összekötés
      4. lm_head: logits = x @ lm_head^T
    """
    tokens = np.asarray(tokens, dtype=np.int64)
    T = tokens.shape[0]

    # 1) Beágyazás: token + pozíció (a pozíciókat 0..T-1-ig használjuk)
    embeddings = state_dict['token_embedding'][tokens] + state_dict['position_embedding'][:T]

    # 2) Első RMSNorm a bemeneten (a hossz egységnyire állítása)
    hidden = rmsnorm(embeddings)

    # Kauzális maszk: háromszög, a (t, t') eleme akkor igaz, ha t' <= t.
    # Vagyis az attention a t-edik pozícióban csak a 0..t pozíciókat láthatja.
    causal_mask = np.tril(np.ones((T, T), dtype=bool))

    blocks = []
    for li in range(n_layer):
        block = {}

        # --- a) attention bemenet: RMSNorm + Q/K/V vetítések ------------------
        block['attn_residual'] = hidden                  # a skip-connection bemenete
        normed = rmsnorm(hidden)
        block['attn_normed'] = normed
        query = normed @ state_dict[f'layer{li}.query_w'].T
        key = normed @ state_dict[f'layer{li}.key_w'].T
        value = normed @ state_dict[f'layer{li}.value_w'].T

        # Fejekre bontás: (T, n_embd) -> (n_head, T, head_dim)
        # A `transpose(1, 0, 2)` a pozíció- és fej-tengelyt cseréli, így a
        # következő matmul fejenként párhuzamosan futhat.
        query_heads = query.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        key_heads = key.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        value_heads = value.reshape(T, n_head, head_dim).transpose(1, 0, 2)
        block['query_heads'], block['key_heads'], block['value_heads'] = (
            query_heads, key_heads, value_heads)

        # --- b) attention számítás -------------------------------------------
        # scores = Q @ K^T / sqrt(head_dim): (n_head, T, T)
        # A sqrt-skalázás miatt a logitek szórása a dimenzió növekedésével nem
        # nő, így a softmax nem "telítődik" 0/1 közeliekre.
        attention_scores = query_heads @ key_heads.transpose(0, 2, 1) / head_dim ** 0.5
        attention_scores = np.where(causal_mask, attention_scores, -np.inf)
        attention_weights = softmax(attention_scores, axis=-1)  # soronként normált
        block['attention_weights'] = attention_weights

        # Minden fej a saját value-vektorainak súlyozott összegét adja.
        head_outputs = attention_weights @ value_heads   # (n_head, T, head_dim)
        heads_combined = head_outputs.transpose(1, 0, 2).reshape(T, n_embd)
        block['heads_combined'] = heads_combined

        # --- c) output vetítés + első residual összekötés ---------------------
        attention_out = heads_combined @ state_dict[f'layer{li}.output_w'].T
        block['attention_out'] = attention_out
        hidden = attention_out + block['attn_residual']  # <-- residual!
        block['after_attention'] = hidden

        # --- d) MLP blokk + második residual összekötés -----------------------
        block['mlp_residual'] = hidden                   # az MLP skip-connection bemenete
        normed = rmsnorm(hidden)
        block['mlp_normed'] = normed
        pre_activation = normed @ state_dict[f'layer{li}.fc1_w'].T   # tágítás 4x-re
        block['pre_activation'] = pre_activation
        activated = np.maximum(0, pre_activation)                    # ReLU
        block['activated'] = activated
        mlp_out = activated @ state_dict[f'layer{li}.fc2_w'].T       # vissza n_embd-re
        block['mlp_out'] = mlp_out
        hidden = mlp_out + block['mlp_residual']         # <-- residual!

        block['block_out'] = hidden
        blocks.append(block)

    # 4) lm_head: a rejtett rétegből a szókészletre vetítünk -> logitek
    logits = hidden @ state_dict['lm_head'].T
    return logits, {'tokens': tokens, 'embeddings': embeddings,
                    'pre_norm': blocks[0]['attn_residual'], 'blocks': blocks}


# -----------------------------------------------------------------------------
# 6) Visszaterjesztés (backward): a kézzel levezetett láncszabály
# -----------------------------------------------------------------------------
# A forward folyamán elmentett értékek (cache) segítségével a modell végéről
# visszafelé haladva kiszámítjuk, mekkora hatással volt a loss minden egyes
# paraméterre (ez a gradiens). Minden művelet gradiense az alábbi szabályokból
# jön:
#   - lineáris réteg:  y = x @ W^T   =>   dW = dY^T @ x ,  dx = dY @ W
#   - softmax backward: dS = w * (dOut - sum(dOut * w))   (w = softmax(S))
#   - RMSNorm backward: ld. a _rmsnorm_backward docstring-jét
#   - cross-entropy:   loss = -log p[target]   =>   dlogits = probs, majd -1
#                      a cél pozíción (a softmax gradiens egybeolvasztva)

def backward(cache, grad_logits, gradients):
    """Visszaterjeszti `grad_logits`-ot a cache-ben tárolt forward értékeken.

    Bemenetek:
        - cache       : a forward() által visszaadott köztes értékek
        - grad_logits : a loss gradiense a logitek szerint (T, vocab_size)
        - gradients   : dict, ahová *hozzáadjuk* (+=) a paraméter-gradienseket

    Az összeadás azért `+=`, mert egyetlen dokumentumon belül a loss a
    pozíciók átlaga, és minden lépés után nullázzuk a gradients dictet.
    """
    blocks = cache['blocks']

    # A legutolsó réteg kimenetére ható gradiens megy az lm_head felé:
    #   logits = hidden @ lm_head^T  =>  dlm_head = grad_logits^T @ hidden
    last_hidden = blocks[-1]['block_out']
    grad_hidden = grad_logits @ state_dict['lm_head']
    gradients['lm_head'] += grad_logits.T @ last_hidden

    # Rétegenként visszafelé (az utolsó rétegtől az első felé)
    for li in reversed(range(n_layer)):
        block = blocks[li]
        T = block['query_heads'].shape[1]

        # ---- MLP ág ----------------------------------------------------------
        # hidden = mlp_out + residual ; innen két út ágazik szét:
        #   a) mlp_out (a blokk belsejébe)
        #   b) a residual skip-connection (közvetlenül továbbmegy)
        grad_mlp_out = grad_hidden

        # mlp_out = activated @ fc2^T  =>  dactivated = dmlp_out @ fc2
        grad_activated = grad_mlp_out @ state_dict[f'layer{li}.fc2_w']
        gradients[f'layer{li}.fc2_w'] += grad_mlp_out.T @ block['activated']

        # ReLU: dpre = dactivated, de csak a pozitív helyeken
        grad_pre_activation = grad_activated * (block['pre_activation'] > 0)
        gradients[f'layer{li}.fc1_w'] += grad_pre_activation.T @ block['mlp_normed']
        grad_mlp_normed = grad_pre_activation @ state_dict[f'layer{li}.fc1_w']

        # RMSNorm visszaterjesztése az MLP bemenetére
        grad_mlp_residual = _rmsnorm_backward(
            block['mlp_normed'], block['mlp_residual'], grad_mlp_normed)

        # A két út összeadódik: ez lesz az attention utáni hidden gradiense
        grad_after_attention = grad_hidden + grad_mlp_residual

        # ---- Attention ág ----------------------------------------------------
        # hidden = attention_out + residual ; megint két út
        grad_attention_out = grad_after_attention
        grad_residual = grad_after_attention

        # attention_out = heads_combined @ output_w^T
        grad_heads_combined = grad_attention_out @ state_dict[f'layer{li}.output_w']
        gradients[f'layer{li}.output_w'] += grad_attention_out.T @ block['heads_combined']
        grad_head_outputs = grad_heads_combined.reshape(T, n_head, head_dim).transpose(1, 0, 2)

        # head_outputs = attention_weights @ value_heads
        #   dattention_weights = dhead_outputs @ value_heads^T
        #   dvalue_heads       = attention_weights^T @ dhead_outputs
        grad_attention_weights = grad_head_outputs @ block['value_heads'].transpose(0, 2, 1)
        grad_value_heads = block['attention_weights'].transpose(0, 2, 1) @ grad_head_outputs

        # softmax backward:
        #   dScores = w * (dW - sum(dW * w))   (w = attention_weights, soronként)
        # A maszkolt (jövőbeli) pozíciókban w = 0, így ott a gradiens is 0 lesz.
        grad_scores = block['attention_weights'] * (
            grad_attention_weights
            - (grad_attention_weights * block['attention_weights']).sum(axis=-1, keepdims=True))

        # scores = Q @ K^T / sqrt(d)  =>  dQ = dScores @ K / sqrt(d) , dK = dScores^T @ Q / sqrt(d)
        scale = head_dim ** 0.5
        grad_query_heads = (grad_scores @ block['key_heads']) / scale
        grad_key_heads = (grad_scores.transpose(0, 2, 1) @ block['query_heads']) / scale

        # Visszarendezés fejekből teljes dimenzióra, majd a Q/K/V vetítések gradiensei
        grad_query = grad_query_heads.transpose(1, 0, 2).reshape(T, n_embd)
        grad_key = grad_key_heads.transpose(1, 0, 2).reshape(T, n_embd)
        grad_value = grad_value_heads.transpose(1, 0, 2).reshape(T, n_embd)

        grad_attn_normed = (grad_query @ state_dict[f'layer{li}.query_w']
                            + grad_key @ state_dict[f'layer{li}.key_w']
                            + grad_value @ state_dict[f'layer{li}.value_w'])
        gradients[f'layer{li}.query_w'] += grad_query.T @ block['attn_normed']
        gradients[f'layer{li}.key_w'] += grad_key.T @ block['attn_normed']
        gradients[f'layer{li}.value_w'] += grad_value.T @ block['attn_normed']

        # Az attention-bemeneti RMSNorm visszaterjesztése + a residual összeadás
        grad_block_input = grad_residual + _rmsnorm_backward(
            block['attn_normed'], block['attn_residual'], grad_attn_normed)
        grad_hidden = grad_block_input

    # Az első RMSNorm után: a gradiens a beágyazásokra száll tovább
    grad_embeddings = _rmsnorm_backward(cache['pre_norm'], cache['embeddings'], grad_hidden)

    # A beágyazási mátrixok: csak az érintett sorokat frissítjük.
    #   token_embedding[tokens] += grad_embeddings   (az használt token id-kre)
    #   position_embedding[:T]  += grad_embeddings   (az 0..T-1 pozíciókra)
    gradients['token_embedding'][cache['tokens']] += grad_embeddings
    gradients['position_embedding'][:cache['tokens'].shape[0]] += grad_embeddings


# -----------------------------------------------------------------------------
# 7) Adam optimalizáló + tréning ciklus
# -----------------------------------------------------------------------------
# Adam két "puffert" vezet be minden paraméterhez:
#   - m : a gradiens mozgóátlaga (első momentum)  -> a "lassú" átlagos irány
#   - v : a gradiens négyzetének mozgóátlaga (második momentum) -> a lépték
# A tanulási sebességet lineárisan csökkentjük (learning rate decay): a tréning
# elején nagy lépéseket tesz, a végén finomakat.
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
num_steps = 96000  # hány dokumentumot "tanuljunk meg" egy sorozatban (~3 epoch)


def train(callback=None):
    """Betanítja a modellt, és visszaadja a loss-ok vektorát.

    Minden lépésben:
      1. Kiveszünk egy dokumentumot (nevet) a sorban.
      2. Tokenizáljuk, körbevesszük BOS tokenekkel.
      3. A teljes szekvenciát egyszerre előrefuttatjuk (forward).
      4. Kiszámítjuk a cross-entropy loss-t: az egyes pozíciókban mennyire volt
         bizonytalan a modell abban, hogy mi a *következő* betű.
      5. Visszaterjesztjük a loss gradienst (backward).
      6. Adam-mal frissítjük a paramétereket, majd nullázzuk a gradienseket.

    A loss gradiens a cross-entropy és a softmax "egybeolvasztott" alakja:
        loss = -log p[target]  =>  dlogits = probs ;  dlogits[target] -= 1
    majd az átlagolás miatt osztunk a pozíciók számával.

    Ha megadsz `callback(step, loss)`-t, minden lépés után meghívjuk vele
    az 1-indexelt lépésszámot és az aktuális loss-t. Ebből tudja a webes
    felület élőben követni a tanulást (loss-görbe, embedding-PCA).
    """
    # A momentum-pufferek a paraméterekkel azonos alakú nulla tömbök.
    # A `losses` mindig CPU-numpy tömb (backendtől független), hogy a tesztek
    # és a vizualizációk ne keveredjenek GPU-tömbökkel.
    first_moment = {key: np.zeros_like(p) for key, p in zip(param_keys, params)}
    second_moment = {key: np.zeros_like(p) for key, p in zip(param_keys, params)}
    losses = _np.zeros(num_steps, dtype=np.float64)

    t0 = time.time()  # benchmark
    for step in range(num_steps):
        # --- 1-2) dokumentum kiválasztása és tokenizálása --------------------
        # n = min(block_size, hossz-1): legfeljebb block_size tokent táplálunk
        # be a modellbe; az n-edik után mindig megjósoljuk a következőt.
        doc = docs[step % len(docs)]
        toks = [BOS] + [stoi[ch] for ch in doc] + [BOS]
        n = min(block_size, len(toks) - 1)
        inp = toks[:n]       # a modell bemenete (n darab token)
        tgt = toks[1:n + 1]  # a célok: minden pozícióban a következő token

        # --- 3) előrehaladás és 4) loss -----------------------------
        logits, cache = forward(inp)
        probs = softmax(logits, axis=-1)               # (n, vocab_size)
        idx = np.arange(n)
        loss = -np.log(probs[idx, tgt]).mean()         # átlagos cross-entropy
        loss_val = float(loss)  # a GPU-s skalar is CPU-float lesz
        losses[step] = loss_val
        if callback is not None:
            callback(step + 1, loss_val)

        # --- 5) visszaterjesztés ------------------------------
        gradients = {key: np.zeros_like(p) for key, p in zip(param_keys, params)}
        grad_logits = probs.copy()
        grad_logits[idx, tgt] -= 1.0
        grad_logits /= n
        backward(cache, grad_logits, gradients)

        # --- 6) Adam frissítés ------------------------------
        lr_t = learning_rate * (1 - step / num_steps)  # lineáris lr-csökkenés
        for key in param_keys:
            param, grad = state_dict[key], gradients[key]
            first_moment[key] = beta1 * first_moment[key] + (1 - beta1) * grad
            second_moment[key] = beta2 * second_moment[key] + (1 - beta2) * grad * grad
            # Torzítatlan becslés a korai lépésekhez (m/ v nulláról indul)
            m_hat = first_moment[key] / (1 - beta1 ** (step + 1))
            v_hat = second_moment[key] / (1 - beta2 ** (step + 1))
            param -= lr_t * m_hat / (np.sqrt(v_hat) + eps_adam)

        if (step + 1) % 100 == 0 or step == 0:
            print(f"step {step+1:4d} / {num_steps:4d} | loss {loss_val:.4f}")

    elapsed = time.time() - t0
    print(f"trained {num_steps} steps in {elapsed:.2f}s ({elapsed / num_steps * 1000:.1f} ms/step)")
    return losses


# -----------------------------------------------------------------------------
# 8) Inferencia: `generate()` — új nevek gyártása
# -----------------------------------------------------------------------------
# A tréning után a modell "motyog": a BOS tokentől (és opcionálisan egy megadott
# előtagtól) indulva tokenenként mintavételez, amíg BOS-t nem kap. A
# `temperature` szabályozza a kreativitást: alacsony értéknél a legvalószínűbb
# következő betűt szinte mindig kiválasztja (konzervatív), magasnál több
# véletlenszerűséget enged.
def generate(prefix='', temperature=0.5, max_len=block_size):
    """Egy nevet generál, opcionálisan egy adott `prefix` folytatásaként.

    Lépésenként betápláljuk a legutóbbi tokent a modellbe, az ekkor keletkező
    query/key/value vektorokat hozzáfűzzük a futó kontextushoz (ez a régebbi,
    pozíció-loopos megoldás inferenciára tökéletesen megfelel), majd a logitek
    softmax-ából mintavételezünk.
    """
    def step(tok, pos, keys, values):
        """Egyetlen token előrehaladása a futó kontextus birtokában."""
        hidden = state_dict['token_embedding'][tok] + state_dict['position_embedding'][pos]
        hidden = rmsnorm(hidden)
        for li in range(n_layer):
            residual = hidden
            normed = rmsnorm(hidden)
            query = normed @ state_dict[f'layer{li}.query_w'].T
            key = normed @ state_dict[f'layer{li}.key_w'].T
            value = normed @ state_dict[f'layer{li}.value_w'].T
            keys[li].append(key)     # a jelenlegi token bekerül a kontextusba
            values[li].append(value)
            context_len = len(keys[li])
            key_stack = np.stack(keys[li])
            value_stack = np.stack(values[li])
            # Fejesített bontás, ahogy a forward-ban is tettük
            query_head = query.reshape(n_head, head_dim)[:, None, :]   # (n_head, 1, head_dim)
            key_heads = key_stack.reshape(context_len, n_head, head_dim).transpose(1, 0, 2)
            value_heads = value_stack.reshape(context_len, n_head, head_dim).transpose(1, 0, 2)
            scores = (query_head @ key_heads.transpose(0, 2, 1)).squeeze(1) / head_dim ** 0.5
            weights = softmax(scores, axis=-1)
            head_out = (weights[:, :, None] * value_heads).sum(axis=1)
            heads_combined = head_out.reshape(n_embd)
            hidden = (heads_combined @ state_dict[f'layer{li}.output_w'].T) + residual
            residual = hidden
            normed = rmsnorm(hidden)
            activated = np.maximum(0, normed @ state_dict[f'layer{li}.fc1_w'].T)
            hidden = (activated @ state_dict[f'layer{li}.fc2_w'].T) + residual
        return hidden @ state_dict['lm_head'].T

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    tokens = [BOS] + [stoi[ch] for ch in prefix]

    # Először betápláljuk a BOS-t és az előtagot, hogy a kontextus feltöltődjön
    logits = None
    for pos, tok in enumerate(tokens):
        logits = step(tok, pos, keys, values)

    # Ezután tokenenként mintavételezünk, amíg BOS-t vagy a hossz-limit el nem érjük
    sample = list(prefix)
    pos = len(tokens)
    while pos < max_len:
        probs = softmax(logits / temperature)
        # size=1: a cupy (GPU) nem támogatja a "size nélküli" choice()-t,
        # a numpy viszont igen — ez a forma mindkét backenden működik.
        tok = int(np.random.choice(vocab_size, size=1, p=probs)[0])
        if tok == BOS:
            break
        sample.append(uchars[tok])
        logits = step(tok, pos, keys, values)
        pos += 1
    return ''.join(sample)


if __name__ == "__main__":
    train()

    # Inferencia: hadd motyogjon a modell új (hallucinált) neveket
    temperature = 0.5  # (0, 1] között; alacsony = kiszámíthatóbb, magas = kreatívabb
    print("\n--- inference (new, hallucinated names) ---")
    for sample_idx in range(20):
        print(f"sample {sample_idx+1:2d}: {generate('', temperature)}")
