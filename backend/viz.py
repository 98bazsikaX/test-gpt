"""
=============================================================================
 microgpt vizualizációk (Plotly)
=============================================================================

Ez a modul a betanított microgpt modellt *értelmezhetővé* teszi. Minden
függvény egy Plotly figure-t ad vissza, amit a webes felület (`name_ui.py`)
kiszolgál, így a böngészőben interaktívan lehet nézegetni (zoom, hover).

Elérhető vizualizációk:
  1) loss_fig()          : a tréning loss-görbéje (nyers + mozgóátlag)
  2) nlayer_fig()        : 1/2/4 rétegű modell loss-görbéi összehasonlítva
  3) pca_token_fig()     : a karakterek token-embeddingjének PCA-ja (2D)
  4) pca_position_fig()  : a pozíció-embeddingek PCA-ja (2D)
  5) attention_fig(név)  : attention hőtérkép fejenként egy adott névre
  6) distribution_fig()  : a "következő betű" valószínűség-eloszlása

Fontos elv: az eldobható tréningek (`loss_fig`, `nlayer_fig`) NEM bántják a
UI-ban használt, cache-ből betöltött modellt — mindig elmentjük az aktuális
állapotot, és a tréning után visszaállítjuk. Ehhez a microgpt.set_config()
és a lentebbi _snapshot/_restore páros szolgál.
"""

from math import ceil

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import microgpt
from microgpt import BOS, block_size, softmax, stoi, uchars

# A plotok a frontend "napfényes papír" témáját követik (ld. frontend/style.css)
NOTEBOOK = go.layout.Template()
NOTEBOOK.layout.paper_bgcolor = '#f6f0e4'
NOTEBOOK.layout.plot_bgcolor = '#fffdf8'
NOTEBOOK.layout.font = {'family': 'system-ui, sans-serif', 'size': 13, 'color': '#1c2433'}
NOTEBOOK.layout.title = {'font': {'family': 'Space Grotesk, sans-serif',
                                  'size': 16, 'color': '#1c2433'}}
NOTEBOOK.layout.xaxis = {'gridcolor': '#e8dfd0', 'zerolinecolor': '#e8dfd0', 'linecolor': '#d6c9b2'}
NOTEBOOK.layout.yaxis = {'gridcolor': '#e8dfd0', 'zerolinecolor': '#e8dfd0', 'linecolor': '#d6c9b2'}
NOTEBOOK.layout.colorway = ['#2f6f68', '#c07a12', '#5b6574', '#24549c', '#8a5a44']

# A tréningek a microgpt.MODEL_LOCK alatt futnak, hogy az eldobható tréningek
# (amik a modul-globálokat írják) ne keveredjenek egy futó UI-tanítással.


# -----------------------------------------------------------------------------
# A modell-állapot mentése / visszaállítása
# -----------------------------------------------------------------------------
def _snapshot():
    """Elmenti a modul globális modell-állapotát (a visszaállításhoz)."""
    return {
        'n_layer': microgpt.n_layer,
        'n_embd': microgpt.n_embd,
        'n_head': microgpt.n_head,
        'block_size': microgpt.block_size,
        'head_dim': microgpt.head_dim,
        'param_keys': microgpt.param_keys,
        'shapes': microgpt.shapes,
        'num_steps': microgpt.num_steps,
        'state_dict': microgpt.state_dict,
        'params': microgpt.params,
    }


def _restore(snap):
    """Visszaállítja a modul globális modell-állapotát `_snapshot()` alapján."""
    microgpt.n_layer = snap['n_layer']
    microgpt.n_embd = snap['n_embd']
    microgpt.n_head = snap['n_head']
    microgpt.block_size = snap['block_size']
    microgpt.head_dim = snap['head_dim']
    microgpt.param_keys = snap['param_keys']
    microgpt.shapes = snap['shapes']
    microgpt.num_steps = snap['num_steps']
    microgpt.state_dict = snap['state_dict']
    microgpt.params = snap['params']


def _train_throwaway(steps, layers=None):
    """Egy eldobható modellt tanít `steps` lépésig, majd visszaállítja az állapotot.

    Ha `layers` meg van adva, előtte átállítja a mélységet (microgpt.set_config).
    A `losses` vektort adja vissza.
    """
    snap = _snapshot()
    try:
        if layers is not None:
            microgpt.set_config(layers)
        else:
            microgpt.reset_params()
        microgpt.num_steps = steps
        return microgpt.train()
    finally:
        _restore(snap)


# -----------------------------------------------------------------------------
# Segédfüggvények
# -----------------------------------------------------------------------------
def _moving_average(values, window=25):
    """Egyszerű mozgóátlag (széleknél a `window`-nál rövidebb átlaggal)."""
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='same')
    # a széleknél a konvolúció 0-val "párosít", ezért ott az eredeti értéket használjuk
    edge = window // 2
    smoothed[:edge] = values[:edge]
    smoothed[-edge:] = values[-edge:]
    return smoothed


def _cpu(x):
    """Egy tömböt CPU-numpy-ra konvertál.

    A tanítás futhat GPU-n (cupy), de a plotly és a numpy-API CPU-t igényel —
    ezért a vizualizációk mindig CPU-másolaton dolgoznak.
    """
    return x.get() if hasattr(x, 'get') else x


def _pca(rows, n_components=2):
    """PCA SVD-vel, numpy-ból — nincs szükség scikit-learnre.

    A középértéket levonjuk, majd az X @ V^T projekció adja az első
    `n_components` főkomponenst.
    """
    rows = _cpu(rows)
    centered = rows - rows.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ Vt[:n_components].T


# -----------------------------------------------------------------------------
# 1) + 2) Tréning-progresszió
# -----------------------------------------------------------------------------
def loss_fig(steps=1000):
    """A tréning loss-görbéje: nyers értékek + mozgóátlag (EMA-szerű)."""
    with microgpt.MODEL_LOCK:
        losses = _train_throwaway(steps)
    x = np.arange(1, len(losses) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=losses, mode='lines', name='loss',
                             line=dict(color='#1f77b4')))
    fig.add_trace(go.Scatter(x=x, y=_moving_average(losses), mode='lines',
                             name='mozgóátlag', line=dict(color='#d62728', dash='dot')))
    fig.update_layout(title=f'Tréning loss ({steps} lépés)',
                      xaxis_title='lépés', yaxis_title='loss',
                      template=NOTEBOOK, height=380)
    return fig


def nlayer_fig(steps=1500):
    """1/2/4 rétegű modellek loss-görbéi ugyanazon a lépésszámon."""
    fig = go.Figure()
    for layers in (1, 2, 4):
        with microgpt.MODEL_LOCK:
            losses = _train_throwaway(steps, layers=layers)
        x = np.arange(1, len(losses) + 1)
        fig.add_trace(go.Scatter(x=x, y=_moving_average(losses), mode='lines',
                                 name=f'{layers} réteg'))
    fig.update_layout(title=f'Rétegszám-összehasonlítás ({steps} lépés)',
                      xaxis_title='lépés', yaxis_title='loss (mozgóátlag)',
                      template=NOTEBOOK, height=380)
    return fig


# -----------------------------------------------------------------------------
# 3) + 4) Beágyazások PCA-ja
# -----------------------------------------------------------------------------
def pca_token_fig():
    """A 27 token-embedding (26 betű + BOS) 2D-s PCA-ja, feliratozva."""
    embeddings = _cpu(microgpt.state_dict['token_embedding'])  # (vocab_size, n_embd)
    labels = uchars + ['BOS']
    proj = _pca(embeddings)
    fig = go.Figure(go.Scatter(
        x=proj[:, 0], y=proj[:, 1], mode='markers+text',
        text=labels, textposition='top center',
        marker=dict(size=10),
    ))
    fig.update_layout(title='Karakter-embeddingek PCA-ja',
                      xaxis_title='1. főkomponens', yaxis_title='2. főkomponens',
                      template=NOTEBOOK, height=380)
    return fig


def pca_position_fig():
    """A 16 pozíció-embedding 2D-s PCA-ja, pozíciószámokkal feliratozva."""
    embeddings = _cpu(microgpt.state_dict['position_embedding'])  # (block_size, n_embd)
    proj = _pca(embeddings)
    fig = go.Figure(go.Scatter(
        x=proj[:, 0], y=proj[:, 1], mode='markers+text',
        text=[str(i) for i in range(embeddings.shape[0])],
        textposition='top center', marker=dict(size=9),
    ))
    fig.update_layout(title='Pozíció-embeddingek PCA-ja',
                      xaxis_title='1. főkomponens', yaxis_title='2. főkomponens',
                      template=NOTEBOOK, height=380)
    return fig


# -----------------------------------------------------------------------------
# 5) Attention hőtérkép
# -----------------------------------------------------------------------------
def _clean_text(text):
    """Kisbetűsíti és megszűri a szöveget a modell ismert karaktereire."""
    return ''.join(c for c in text.lower() if c in stoi)


def attention_fig(name, layer=0):
    """Attention hőtérkép fejenként, egy adott névre.

    A modellen átfuttatjuk a nevet (`microgpt.forward`), és a cache-ből
    kiolvassuk az attention súlyokat: (n_head, T, T) alakú kauzális mátrix.
    A (t, t') cella megmutatja, mennyire figyelt a t-edik betű a t'-edikre.
    """
    name = _clean_text(name)[:block_size - 1]
    if not name:
        raise ValueError('a név nem tartalmaz érvényes betűt (a-z)')

    tokens = [BOS] + [stoi[c] for c in name]
    _, cache = microgpt.forward(tokens)
    weights = _cpu(cache['blocks'][layer]['attention_weights'])  # (n_head, T, T)

    labels = ['BOS'] + list(name)
    heads = microgpt.n_head
    cols = 2
    rows = ceil(heads / cols)
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f'{h}. attention fej' for h in range(heads)])
    for h in range(heads):
        row, col = divmod(h, cols)
        fig.add_trace(
            go.Heatmap(z=weights[h], x=labels, y=labels,
                       colorscale='Viridis', zmin=0, zmax=1,
                       showscale=(h == 0)),
            row=row + 1, col=col + 1,
        )
    fig.update_layout(title=f'Attention súlyok a névre: „{name}”',
                      template=NOTEBOOK, height=380 * rows)
    return fig


# -----------------------------------------------------------------------------
# 6) A "következő betű" valószínűség-eloszlása
# -----------------------------------------------------------------------------
def distribution_fig(prefix):
    """Annak az eloszlása, hogy az adott prefix után milyen betű jöhet.

    A prefixet (BOS-szal) átfuttatjuk a modellen, és a legutolsó pozíció
    logitjeinek softmax-át mutatjuk meg a 26 karakterre.
    """
    prefix = _clean_text(prefix)[:block_size - 1]
    if not prefix:
        raise ValueError('a prefix nem tartalmaz érvényes betűt (a-z)')

    tokens = [BOS] + [stoi[c] for c in prefix]
    logits, _ = microgpt.forward(tokens)
    probs = _cpu(softmax(logits[-1]))  # a következő token valószínűségei

    fig = go.Figure(go.Bar(x=uchars, y=probs, marker=dict(color='#2ca02c')))
    fig.update_layout(title=f'Következő betű eloszlása „{prefix}” után',
                      xaxis_title='következő betű', yaxis_title='valószínűség',
                      template=NOTEBOOK, height=380)
    return fig

