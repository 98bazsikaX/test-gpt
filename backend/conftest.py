"""Pytest gyökérfájl — a tesztek mindig CPU-n (numpy) futnak.

A microgpt GPU-gépen cupy-ra kapcsolhat (BACKEND='gpu'), de a teszteknek
backend-függetleneknek és determinisztikusaknak kell maradniuk (pl. a CI-n
nincs CUDA). Ez az autouse fixture minden teszt előtt CPU-ra állítja a
backendet, és friss paramétereket ad.
"""

import pytest

import microgpt


@pytest.fixture(autouse=True)
def _force_cpu_backend():
    microgpt.set_backend('cpu')
    microgpt.reset_params()
    yield
    microgpt.set_backend('cpu')
