"""A backend Docker-konténer indítója: cache-betöltés + uvicorn rögzített porton.

Helyi fejlesztéshez nem kell — a `name_ui.py` `__main__` blokkja véletlenszerű
porton + böngésző-nyitással indít. Konténerben viszont fix porton futunk,
böngésző-nyitás nélkül.
"""

import uvicorn

import name_ui

if __name__ == '__main__':
    name_ui._load_cache()  # cache betöltése, vagy háttér-tanítás indítása
    uvicorn.run(name_ui.app, host='0.0.0.0', port=8000, log_level='warning')
