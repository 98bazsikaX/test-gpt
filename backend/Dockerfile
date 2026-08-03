# microgpt backend — Python 3.12 + uv
FROM python:3.12-slim

WORKDIR /app

# Függőségek a lockfile-ból (reprodukálható, gyorsítótárazható réteg)
COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-dev

# Csak a futtatáshoz szükséges fájlok (a model.pkl/input.txt futásidőben jön létre)
COPY microgpt.py name_ui.py viz.py server.py ./

EXPOSE 8000
CMD ["uv", "run", "python", "server.py"]
