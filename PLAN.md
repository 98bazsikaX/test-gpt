# PLAN.md — fejlesztési terv

A `my-little-jpa` mintájára készült monorepo-terv: a projekt **backend/** (Python)
és **frontend/** (statikus web) részekre van bontva, docker-compose-zal.

## Állapot (✅ kész)

- [x] A modell numpy-os átirata (Karpathy microgpt alapján), kézi backward-dal
- [x] Tesztek a referencia-autograddal való ekvivalenciára
- [x] FastAPI backend: kiegészítés, temperature, újratanítás (háttér-szál)
- [x] Élő vizualizációk: loss-görbe + embedding-PCA a tanulás közben
- [x] Vizualizáció-galería: rétegszám-összehasonlítás, attention, eloszlás
- [x] Monorepo: `backend/` + `frontend/`, helyi futtatás a backendből
- [x] Docker: `backend/Dockerfile` (uvicorn) + `frontend/Dockerfile` (nginx) + `docker-compose.yml`
- [x] README, LICENSE (MIT), `.editorconfig`, `.opencode/AGENTS.md` (ágens-memória)
- [x] CI: GitHub Actions — `uv run pytest` (3.12/3.13) + `ruff` minden push/PR-re,
      GHCR image-build `main` push után

## Következő lépések (🚧 tervezett)

- [ ] A `model.pkl` / `input.txt` docker-én való perzisztenciája (named volume)
- [ ] HuggingFace-token helyett opcionális súly-export (safetensors/pytorch) a
      tanult modellhez
- [ ] A `frontend` egyedi favicon és jobb responszív elrendezés
- [ ] A `complete_name.py` cache-elése (most minden futáskor újratanít)

## Mérföldkövek

| Mérföldkő | Tartalom | Állapot |
|---|---|---|
| M0 | Modell mag + referencia-tesztek | ✅ |
| M1 | Webes felület + vizualizációk | ✅ |
| M2 | Monorepo + Docker | ✅ |
| M3 | Perzisztencia | 🚧 |
| M4 | CI (ruff + pytest + GHCR) | ✅ |
