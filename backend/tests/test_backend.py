import numpy as np
import pytest

import microgpt


def test_backend_is_cpu_in_tests():
    """A conftest CPU-ra kényszerít — a tesztek nem függhetnek GPU-tól."""
    assert microgpt.BACKEND == 'cpu'


def test_set_backend_validates():
    microgpt.set_backend('cpu')
    assert microgpt.BACKEND == 'cpu'
    with pytest.raises(ValueError):
        microgpt.set_backend('bogus')


def test_train_returns_cpu_losses():
    """A `train()` a losses-t mindig CPU-numpy tömbként adja (backendtől függetlenül)."""
    microgpt.configure(layers=1, embd=16, heads=4, block=16, fresh=True)
    microgpt.num_steps = 20
    losses = microgpt.train()
    assert isinstance(losses, np.ndarray)
    assert losses.dtype == np.float64


def test_load_weights_roundtrip_active_backend():
    """A cache-betöltés után a súlyok az aktív backendre kerülnek."""
    microgpt.configure(layers=2, embd=16, heads=4, block=16, fresh=True)
    saved = {k: microgpt._to_cpu(v).copy() for k, v in microgpt.state_dict.items()}
    microgpt.load_weights(saved, 2, 16, 4, 16)
    for key in microgpt.param_keys:
        np.testing.assert_array_equal(microgpt._to_cpu(microgpt.state_dict[key]), saved[key])


def test_gpu_path_when_available():
    """GPU-gépen: a tréning cupy-n fut, de a losses továbbra is CPU-numpy."""
    if not microgpt._GPU_AVAILABLE:
        pytest.skip('nincs CUDA a gépen')
    try:
        microgpt.set_backend('gpu')
        microgpt.reset_params()
        microgpt.configure(layers=1, embd=16, heads=4, block=16, fresh=True)
        microgpt.num_steps = 10
        losses = microgpt.train()
        assert microgpt.BACKEND == 'gpu'
        assert losses.dtype == np.float64
        assert hasattr(microgpt.state_dict['token_embedding'], 'device')  # cupy tömb
    finally:
        microgpt.set_backend('cpu')
        microgpt.reset_params()
