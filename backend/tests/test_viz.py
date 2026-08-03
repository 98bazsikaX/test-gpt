import pickle

import numpy as np
import plotly.graph_objects as go
import pytest

import microgpt
import viz


def _small_config():
    microgpt.configure(layers=1, embd=16, heads=4, block=16, fresh=True)


def test_configure_sets_dimensions():
    microgpt.configure(layers=2, embd=32, heads=8, block=8, fresh=True)
    assert microgpt.n_layer == 2
    assert microgpt.n_embd == 32
    assert microgpt.n_head == 8
    assert microgpt.block_size == 8
    assert microgpt.head_dim == 4
    assert len(microgpt.param_keys) == 3 + 6 * 2
    # az újonnan épített súlyok alakjai illeszkednek a konfigurációhoz
    assert microgpt.state_dict['token_embedding'].shape == (microgpt.vocab_size, 32)
    assert microgpt.state_dict['layer1.query_w'].shape == (32, 32)


def test_configure_rejects_bad_heads():
    with pytest.raises(ValueError):
        microgpt.configure(layers=1, embd=16, heads=3, block=8, fresh=True)
    with pytest.raises(ValueError):
        microgpt.configure(layers=1, embd=16, heads=4, block=1, fresh=True)


def test_load_weights_roundtrip():
    microgpt.configure(layers=2, embd=32, heads=8, block=8, fresh=True)
    data = pickle.dumps({
        'n_layer': microgpt.n_layer, 'n_embd': microgpt.n_embd,
        'n_head': microgpt.n_head, 'block_size': microgpt.block_size,
        'state_dict': microgpt.state_dict,
    })
    saved = pickle.loads(data)
    copies = {k: v.copy() for k, v in saved['state_dict'].items()}

    microgpt.configure(fresh=True)  # más konfigurációra ugrunk
    microgpt.load_weights(saved['state_dict'], saved['n_layer'], saved['n_embd'],
                          saved['n_head'], saved['block_size'])
    assert microgpt.n_layer == 2
    assert microgpt.n_embd == 32
    assert microgpt.n_head == 8
    for key in microgpt.param_keys:
        np.testing.assert_array_equal(microgpt.state_dict[key], copies[key])


def test_train_callback_reports_each_step():
    np.random.seed(42)
    microgpt.configure(layers=1, embd=16, heads=4, block=16, fresh=True)
    microgpt.num_steps = 25
    calls = []
    microgpt.train(callback=lambda step, loss: calls.append((step, loss)))
    assert len(calls) == 25
    assert calls[0][0] == 1
    assert calls[-1][0] == 25
    assert all(np.isfinite(loss) for _, loss in calls)


def test_throwaway_training_restores_model():
    microgpt.configure(layers=2, embd=32, heads=8, block=8, fresh=True)
    microgpt.num_steps = 20
    np.random.seed(42)
    microgpt.reset_params()
    microgpt.train()
    before = {k: v.copy() for k, v in microgpt.state_dict.items()}
    before_config = (microgpt.n_layer, microgpt.n_embd, microgpt.n_head, microgpt.block_size)

    viz._train_throwaway(15)  # eldobható tréning, pl. rétegösszehasonlítás

    for key in before:
        np.testing.assert_array_equal(microgpt.state_dict[key], before[key])
    current = (microgpt.n_layer, microgpt.n_embd, microgpt.n_head, microgpt.block_size)
    assert current == before_config


def test_loss_fig_has_raw_and_smoothed():
    _small_config()
    fig = viz.loss_fig(steps=40)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_nlayer_fig_has_three_traces():
    _small_config()
    fig = viz.nlayer_fig(steps=40)
    assert len(fig.data) == 3


def test_pca_figs_shapes():
    _small_config()
    fig = viz.pca_token_fig()
    assert len(fig.data[0].x) == microgpt.vocab_size
    fig2 = viz.pca_position_fig()
    assert len(fig2.data[0].x) == microgpt.block_size


def test_attention_fig_per_head():
    _small_config()
    fig = viz.attention_fig('karla')
    assert len(fig.data) == microgpt.n_head  # minden fejhez egy hőtérkép
    with pytest.raises(ValueError):
        viz.attention_fig('')


def test_distribution_fig():
    _small_config()
    fig = viz.distribution_fig('ka')
    assert len(fig.data[0].x) == 26  # a 26 karakter
    with pytest.raises(ValueError):
        viz.distribution_fig('!!')
