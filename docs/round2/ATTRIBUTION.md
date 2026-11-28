# Attribution: every third-party component, its licence and its role

Round 2, 14 November 2026. Built by reading the files that actually pin the dependencies (`pipeline/pyproject.toml`,
`pipeline/scripts/install_gpu.sh`, `pipeline/Dockerfile`, `pipeline/wheels.lock.json`, `app/package.json`,
`cloud/deploy.sh`, `NOTICE`) and then checking each licence against its source, not from memory.

Three levels of confidence appear below. **Read** means the licence text was fetched and read on 14 November 2026.
**Registry** means the licence field came from the publisher's own metadata (GitHub's licence API or the npm registry)
on the same day. **Vendored** means the licence text ships inside this repository. Anything without one of those is marked unverified in section 8.

Ply itself is proprietary: copyright (c) 2026 Ricco Marcelino, all rights reserved, see `LICENSE`. `NOTICE` carries the
copyright lines. Nothing below narrows those licences, and nothing in Ply's terms applies to them.

## 1. Pipeline and GPU image

| Component | Version or pin | Licence | Role | Evidence |
|---|---|---|---|---|
| COLMAP, through pycolmap-cuda12 | 4.2.1 | BSD-3-Clause | structure from motion, matching | Read: COLMAP `LICENSE` |
| gsplat | 1.6.0, pin `512d366` | Apache-2.0 | the splat trainer, MCMC and antialiasing | Registry: gsplat repo |
| fused-ssim | pin `a7c48d6` | MIT | the SSIM term in training | Registry: rahul-goel/fused-ssim |
| fused-bilagrid | pin `49f0ef0` | MIT | the bilateral grid the exposure model uses | Registry: harry7557558/fused-bilagrid |
| ppisp | 1.2.1, nv-tlabs | Apache-2.0 | per-photo exposure and colour model | Registry: nv-tlabs/ppisp |
| nerfview | pin `4538024` | Apache-2.0 | the training-time viewer | Registry: nerfstudio-project/nerfview |
| PyTorch, torchvision | 2.11.0+cu128, 0.26.0 | BSD-3-Clause | tensors, CUDA runtime for everything above | Read: PyTorch `LICENSE` |
| numpy | 2.5.4 | BSD-3-Clause | arrays, every stage | Read: numpy `LICENSE.txt` |
| scipy | >= 1.11 | BSD-3-Clause | nearest-point search in the LiDAR path | Registry: scipy repo |
| OpenCV, headless wheel | 5.0.0.93 pinned, >= 4.9 asked | Apache-2.0 | image I/O, undistortion, the occupancy image | Registry: opencv repo |
| Pillow | 12.3.0 | MIT-CMU (the PIL licence) | image I/O for the labeller | Read: Pillow `LICENSE` |
| Hugging Face transformers | 5.19.0 | Apache-2.0 | runs SAM 3 for the label stage | Registry: transformers repo |
| accelerate | >= 1.1.0 | Apache-2.0 | device placement for SAM 3 | Registry: transformers repo |
| MoGe-2, utils3d | pins `0744441`, `3fab839` | MIT | opt-in metric depth prior | Read: MoGe `LICENSE`; Registry: utils3d |
| trimesh | >= 4, test extra only | **unverified** | `pipeline/tests/test_mesh_fusion.py`, not imported by the worker | no licence file found |
| modal client | 1.6.1 | Apache-2.0 | launching GPU jobs | Registry: modal-labs/modal-client |
| boto3 | 1.43.111 | Apache-2.0 | writing results to Backblaze B2 | Registry: boto/boto3 |
| huggingface_hub | 2.2.0 | Apache-2.0 | publishing twins | Registry: huggingface repo |
| kaggle | 2.2.4 | **unverified** | pushing the Kaggle notebook | no licence file found by the registry |
| ffmpeg | Ubuntu 24.04 package | LGPL-2.1-or-later project-wide, **build variant unverified** | decoding clips, extracting frames | Read: FFmpeg `LICENSE.md` |
| Node.js | 22.14.0 | MIT | runs `splat-transform` inside the image | Read: Node `LICENSE` |
| `@playcanvas/splat-transform` | 3.10.1, run by `npx` | MIT | writes the SPZ the viewer streams | Registry: playcanvas/splat-transform |
| CUDA toolkit, cuDNN | 12.8, from `nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04` | NVIDIA proprietary EULA, redistributable parts only | compiles the CUDA extensions | vendor terms, not a repository licence |
| `nvidia/cuda` container image | 12.8.1 | NVIDIA Deep Learning Container licence | the worker image's base | vendor terms |

Versions marked "pinned" also appear in `pipeline/wheels.lock.json`, which records the four CUDA wheels this project
built and the SHA-256 of each (`gsplat 1.6.0`, `fused-ssim 1.0.0`, `fused-bilagrid 0.0.0`, `ppisp 1.2.1`). Transitive
Python packages are installed from PyPI under their own licences; `pipeline/requirements.frozen-backup.txt` is the
record of what a working environment contained, and it is not the worker image's list.

## 2. Web application and build

| Component | Version | Licence | Role | Evidence |
|---|---|---|---|---|
| three.js | 0.186.1 | MIT | the renderer | Registry: npm |
| `@sparkjsdev/spark` | 2.3.1 | MIT | splat streaming and rendering | Registry: npm |
| React, react-dom | 19.3.0 | MIT | the UI | Registry: npm |
| marked | 18.1.1 | MIT | renders the capture guide from Markdown | Registry: npm |
| qrcode | 1.5.4 | MIT | the marker's QR code | Registry: npm |
| Vite | 8.3.4 | MIT | build and dev server | Registry: npm |
| TypeScript | 7.0.2 | Apache-2.0 | types | Registry: npm |
| Tailwind CSS, `@tailwindcss/vite` | 4.3.3 | MIT | styles | Registry: npm |
| `@vitejs/plugin-react`, `@vitejs/plugin-basic-ssl` | 6.1.2, 2.3.0 | MIT | build plugins | Registry: npm |
| `@types/three`, `@types/react`, `@types/react-dom`, `@types/qrcode` | 0.186.0, 19.3.0, 19.3.0, 1.5.6 | MIT | types only | Registry: npm |

The Cloudflare Worker in `cloud/` has **no npm dependencies**: `cloud/wrangler.toml`, `cloud/src/*.ts` and the D1 schema
are the whole service, and the tests run on Node's own test runner. The deploy uses `npx wrangler@4.149.0`, pinned in
`cloud/deploy.sh` and in `.github/workflows/api.yml`; wrangler is Cloudflare's CLI under Apache-2.0, installed at deploy
time rather than vendored.

## 3. Vendored assets

| Asset | Version | Licence | Role | Evidence |
|---|---|---|---|---|
| SuperSplat | v2.32.5 | MIT, copyright PlayCanvas Ltd | the editor served at `/ply/editor/` | Vendored: `app/public/editor/LICENSE.supersplat.txt` |
| Anatomy isometric figure kit | pin as vendored | MIT, copyright (c) 2026 Ryan | the figures in the landing film | Vendored: `app/figures/kit/LICENSE.anatomy.txt` |

SuperSplat is built from tag v2.32.5 with one dependency change, `@playcanvas/splat-transform` 3.3.3 to 3.10.3, so it
reads SPZ v4. That is the only modification and it is disclosed in `NOTICE`.

## 4. Model weights

| Weights | Licence | Role | Redistributed by Ply |
|---|---|---|---|
| SAM 3, `facebook/sam3` and `facebook/sam3.1` | SAM License, last updated 19 November 2025, copyright Meta Platforms | object detection for `places.json` | no: downloaded at image build with a Hugging Face token, never committed |
| MoGe-2 | MIT (repository licence) | opt-in depth prior | no: installed at run time from the pinned commit |

The SAM License permits commercial use, requires the licence to accompany any redistribution of the materials, and
prohibits some end uses. The licence is recorded here and in `NOTICE`, and Ply ships neither the weights nor a
container image that holds them.

## 5. Data

| Data | Licence | Use | Evidence |
|---|---|---|---|
| Chapel capture, 8 clips | Ply's own, owner's footage | the Round 2 twin `chapel-q1b-1` | in the repository's record only |
| Classroom capture, 5 clips | Ply's own | `hq-classroom-1`, Round 1 twin | `test/README.md` |
| Office laser scan, 184.6 M points | Ply's own scan | `lidar-office-8b` | `docs/EXPERIMENTS.md` |
| MuSHRoom honka, 335 photos with a Faro scan | CC-BY-4.0 | scale reference, 5.16% error | Zenodo record 10230733, licence field read |
| MuSHRoom coffee_room with reference mesh | CC-BY-4.0 | scale reference, 3.12% error | Zenodo record 10222321, licence field read |
| Dr Johnson, Deep Blending, 263 photos | **unverified** | cheap experiments only, no round-2 figure depends on it | fetched from a Kaggle copy that now returns 404 |
| Demo bedroom, `app/public/scenes/demo` | **unverified** | the design lab's stand-in room, streamed from Aholo's CDN | no licence recorded anywhere in the repository |

The Demo bedroom is the one to fix before the deadline: its splat is streamed from `holo-cos.aholo3d.cn` and the
repository records it as "a stand-in until our own pipeline output". It is used by the design lab and the bundled demo,
not by any twin in this round's figures, and it should be replaced with a Ply capture or removed from the public build.

## 6. Services

| Service | What it does for Ply | Terms |
|---|---|---|
| Cloudflare Workers, D1, R2 | the API, the job queue, uploads and result serving | Cloudflare developer terms, free tier |
| Modal | GPU workers for training, export and labelling | Modal terms, metered per second |
| Backblaze B2 | S3-compatible bucket for uploads and large results | Backblaze terms |
| Hugging Face Hub | public dataset of twins and their files | Hugging Face terms; dataset visibility is separate |
| Kaggle | the notebook that drives the worker by hand | Kaggle terms |
| GitHub, GitHub Actions, GitHub Pages | source, CUDA wheel builds, the site | GitHub terms |
| Google Chrome | renders the PDFs in this folder with `--headless --print-to-pdf` | Google Chrome terms |

None of these ships code inside Ply. Their client libraries are listed in section 1.

## 7. What Ply contributes

For the judges' originality check, and to keep the credit honest: the pipeline stages (frames, align, occupancy, label,
train launcher, export, gates), the worker, the API, the viewer's walk engine, the router with its clearance tiers, the
place clustering, the search palette, the capture app and this document set are Ply's own code. The third-party tools
above supply the solver, the trainer, the detector's weights, the renderer and the file formats. Where a component is
used unchanged it is listed with its pin; where it is modified, the modification is disclosed (SuperSplat's dependency
bump).

## 8. Not verified, and how to close it

| Item | Why it is not confirmed | To close it |
|---|---|---|
| trimesh licence | GitHub reports no licence file for the repository | read the licence of the installed version, or drop `test_mesh_fusion.py` |
| kaggle client licence | same | it is a development tool, not shipped; state it as a tool or remove it |
| FFmpeg build variant | Ubuntu's build enables GPL components; which ones the image carries was not checked | `ffmpeg -version` in the image prints the configuration; record the flag |
| Dr Johnson dataset terms | the Kaggle copy is gone | use MuSHRoom for anything that has to be published, which is what the scale work already does |
| Demo bedroom capture | no licence recorded | replace with a Ply capture or remove from the public build |
| Pillow's SPDX string | the licence is the MIT-CMU text; "HPND" is the modern SPDX name for it and the file does not use that word | cite it as "MIT-CMU (PIL)" in submission text, as this document does |

## 9. Re-checking this list

```bash
cd pipeline && python3 -m pip show gsplat pycolmap-cuda12 torch transformers modal 2>/dev/null | grep -E "^(Name|Version|License)"
npx --yes @playcanvas/splat-transform@3.10.1 --version
curl -s https://api.github.com/repos/nerfstudio-project/gsplat/license | grep spdx
```

Everything here was checked on 14 November 2026. `NOTICE` in the repository root is the copy that ships with the code,
and it should be updated in the same commit as any new dependency.
