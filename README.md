# Ply

Walk a building once with a phone; get a photoreal, walkable, searchable 3D twin with indoor navigation.
Team Ply · Techfest 2026-27 "Namma Space".

| Folder | What |
|---|---|
| `app/` | The website (Vite + React + three.js + Spark): landing film, in-browser capture app, WASD walkthrough, capture guide. Deployed to https://ricco.codes/ply/ by `.github/workflows/pages.yml`. |
| `cloud/` | Ply API: one Cloudflare Worker (R2 uploads, D1 job queue, results for the viewer). |
| `pipeline/` | `plypipe`: video → frames → COLMAP → floor/scale → occupancy → gsplat → SPZ, plus the GPU worker (`python -m plypipe.worker`) and Kaggle/RunPod setups. |
| `docs/` | Capture SOP (also rendered at /ply/guide.html). |
