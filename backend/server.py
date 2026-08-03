"""A backend Docker-konténer indítója: cache-betöltés + uvicorn.

Helyi fejlesztéshez nem kell — a `name_ui.py` `__main__` blokkja 9000-es porton,
böngésző-nyitás nélkül indít. Konténerben fix porton futunk (MICROGPT_PORT,
alapból 8000), böngésző-nyitás nélkül.

Ha a MICROGPT_PORT-ot megváltoztatod, a frontend/nginx.conf `proxy_pass`-ében
is állítsd át a backend portját (a nginx képe a conf-fal együtt épül).
"""

import os

import uvicorn

import name_ui

if __name__ == '__main__':
    name_ui._load_cache()  # cache betöltése, vagy háttér-tanítás indítása
    port = int(os.environ.get('MICROGPT_PORT', '8000'))
    uvicorn.run(name_ui.app, host='0.0.0.0', port=port, log_level='warning')
