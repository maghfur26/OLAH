---
title: OLAH Recipe Recommender API
emoji: 🍳
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# OLAH Recipe Recommender API

REST API untuk rekomendasi resep masakan Indonesia berbasis Deep Learning.

## Endpoints
- `GET /health` — status model
- `GET /docs` — Swagger UI
- `POST /recommend` — rekomendasi dari bahan
- `POST /similar` — resep mirip
- `GET /recipe/popular` — resep populer
- `GET /recipe/random` — resep random
