# Ply, Round 2: every link in one document

This is the single master document the rules ask for: every supplementary link for the round, one line each, with the
HTTP status it returned when fetched without credentials on 14 November 2026. Nothing here needs an account except where
the status column says so, and those rows are also the to-do list in section 5.

Publish this file in the submission folder as `Ply_Round2_Master_Links.pdf`; the Markdown it is rendered from is next to
it, and `docs/round2/render.py` re-renders it.

## 1. The twins, in a browser

Controls on every twin: click once to take the mouse, W A S D to walk, mouse to look, `Esc` to release. `Ctrl`/`Cmd`+`K`
opens the place search, the minimap in the corner shows the twin from above and opens large when clicked, `Views` jumps
to the capture's own shots, `H` toggles help.

| What | Link | Status |
|---|---|---|
| Ply site, landing film, entry to everything | <https://ricco.codes/ply/> | 200 |
| Twin for this round, `chapel-q1b-1`: 1,223 poses, 281 places, 10 cm walking grid | <https://ricco.codes/ply/walk.html?scene=https%3A%2F%2Fply-api.ricco-ply.workers.dev%2Fv1%2Fscenes%2Fchapel-q1b-1%2Fscene.json> | 200 |
| Fallback twin, `chapel-high-2`: same solve, 60,000 steps, 1.06 M splats | <https://ricco.codes/ply/walk.html?scene=https%3A%2F%2Fply-api.ricco-ply.workers.dev%2Fv1%2Fscenes%2Fchapel-high-2%2Fscene.json> | 200 |
| Classroom twin, `hq-classroom-1`: 542 poses, 194 places, 1.35 M splats | <https://ricco.codes/ply/walk.html?scene=https%3A%2F%2Fply-api.ricco-ply.workers.dev%2Fv1%2Fscenes%2Fhq-classroom-1%2Fscene.json> | 200 |
| LiDAR twin, `lidar-office-8b`: 184.6 M point scan, 1.92 M splats, no photos | <https://ricco.codes/ply/walk.html?scene=https%3A%2F%2Fply-api.ricco-ply.workers.dev%2Fv1%2Fscenes%2Flidar-office-8b%2Fscene.json> | 200 |
| Raw LiDAR cloud, 4.34 M points at 1 cm, streamed as a point cloud | <https://ricco.codes/ply/lidar.html> | 200 |
| Exposure look-test, the two chapel twins side by side on one camera | <https://ricco.codes/ply/ab.html> | 200 |
| Capture app: records, checks the walk and uploads from a phone | <https://ricco.codes/ply/capture.html> | 200 |
| Capture procedure, rendered from the SOP for the phone | <https://ricco.codes/ply/guide.html> | 200 |
| Splat editor, SuperSplat v2.32.5, opened on the capture's own camera | <https://ricco.codes/ply/editor/> | 200 |
| Library of the team's jobs | <https://ricco.codes/ply/library.html> | 200, needs the team code |

The scene files themselves are public and are what the viewer reads. They are worth listing because a judge can open one
and see the twin's contract without the site:

| File | Link | Status |
|---|---|---|
| `chapel-q1b-1` scene manifest | <https://ply-api.ricco-ply.workers.dev/v1/scenes/chapel-q1b-1/scene.json> | 200 |
| `chapel-q1b-1` places, 281 objects with class, confidence and box | <https://ply-api.ricco-ply.workers.dev/v1/scenes/chapel-q1b-1/places.json> | 200 |
| `chapel-q1b-1` occupancy grid, 206 x 87 cells at 10 cm | <https://ply-api.ricco-ply.workers.dev/v1/scenes/chapel-q1b-1/occupancy.png> | 200 |
| `chapel-q1b-1` top view, the minimap, 0.02 m per pixel | <https://ply-api.ricco-ply.workers.dev/v1/scenes/chapel-q1b-1/topview.webp> | 200 |
| API root, which reports the deployed version and time | <https://ply-api.ricco-ply.workers.dev/> | 200 |

## 2. Code, data and sample captures

| What | Link | Status |
|---|---|---|
| Repository, all of it: `app/`, `cloud/`, `pipeline/`, `docs/` | <https://github.com/Ricco-Mallick/ply> | 200, public |
| Sample capture and the run command | <https://github.com/Ricco-Mallick/ply/tree/main/test> | 200 |
| Pipeline setup and the CUDA wheels it pins | <https://github.com/Ricco-Mallick/ply/tree/main/pipeline> | 200 |
| MuSHRoom honka: iPhone walk with a Faro laser scan, our scale reference | <https://zenodo.org/records/10230733> | 200, CC-BY-4.0 |
| MuSHRoom coffee_room: the second laser reference | <https://zenodo.org/records/10222321> | 200, CC-BY-4.0 |
| Ply's published twins and their per-job files | <https://huggingface.co/datasets/Ricco-Marcelino/ply-data> | **401 without a token: not public yet** |

The twin files a visitor loads are served by the API, so the viewer works without the dataset being public. The dataset
link must not go into the submission until the repository is public, and section 5 says what to do about it.

The small scene used for cheap experiments is Dr Johnson from Deep Blending, 263 photos at 1332 x 876, fetched from a
Kaggle copy during the project. That copy returns 404 without a Kaggle login today, so it is not listed as a
deliverable. Nothing in this round's figures depends on a reader having it.

## 3. Documents in this folder

| File | What it is | Pages |
|---|---|---|
| `ABSTRACT.md` | the technical abstract, with the credits section the rules require | 2 |
| `SOP.md` | the capture procedure: device, settings, lighting, movement, angles, distance, sequence, corners, floor plan, examples, raw video, multi-room and multi-floor | 3 |
| `COST-ANALYSIS.md` | measured provider prices against an owned GPU, with break-even and a utilisation table | 3 |
| `ATTRIBUTION.md` | every third-party component with its licence, version or pin, and the evidence | 2 |
| `VIDEO-SCRIPT.md` | the 3 to 5 minute demo: timecoded shots and the words to say, in the owner's voice | 3 |
| `MASTER-PDF.md` | this document | 2 |

The abstract and the SOP are 5 pages together, which is the organisers' limit, and both say so in their own text.
`render.py` prints the page count of every document, and the numbers above are its output.

These sit in `docs/` and are linked from the repository:

| File | What it is |
|---|---|
| `docs/STATUS.md` | what is deployed, what is running, what is next, dated at the end of every session |
| `docs/EXPERIMENTS.md` | every paid run, its numbers and its cost, wins and losses |
| `docs/PROCESSING.md` | how a run plans its own frame rate, resolution and gaussian budget |
| `docs/report.pdf` | the Round 1 technical abstract, five pages, the style these documents are set in |

## 4. Video and slides

| Item | State |
|---|---|
| Round 2 demo video, 3 to 5 minutes, owner's voice | **not recorded yet.** `VIDEO-SCRIPT.md` has the shot list, the exact words per shot and the pre-recording checklist. This row becomes the link on the day it is uploaded, unlisted or public, and it is checked before the folder goes out. |
| Slide deck | none. The final is a 10 minute live pitch and 5 minutes of questions, so there is no deck to ship now: it gets built for the on-site round from `ABSTRACT.md`. If one exists by 30 November it goes here. |

## 5. Before this folder goes out

1. **The Hugging Face dataset is private.** Either make `Ricco-Marcelino/ply-data` public and re-check the row in
   section 2, or delete that row. Every other link in this document already returns 200 without credentials.
2. **The video is not recorded.** Record from `VIDEO-SCRIPT.md`, upload, then put the link in section 4 and re-render
   this PDF.
3. **The library page needs a team code.** Decide whether to publish a code with the submission or drop the row.
4. **Raw capture video is not published.** The SOP lists the clips and where they sit today; publish them under
   `captures/` and fill the SOP's status column before the deadline.
5. **Re-run the check below** and confirm that every row still returns 200 on the day of submission.

## 6. Checking this list again

```bash
# every link in this document, without credentials; 200 is a pass, 401 and 404 are failures
for u in https://ricco.codes/ply/ \
         https://ricco.codes/ply/walk.html \
         https://ricco.codes/ply/capture.html \
         https://ricco.codes/ply/guide.html \
         https://ricco.codes/ply/editor/ \
         https://ricco.codes/ply/ab.html \
         https://ricco.codes/ply/lidar.html \
         https://ply-api.ricco-ply.workers.dev/ \
         https://ply-api.ricco-ply.workers.dev/v1/scenes/chapel-q1b-1/scene.json \
         https://ply-api.ricco-ply.workers.dev/v1/scenes/chapel-q1b-1/places.json \
         https://ply-api.ricco-ply.workers.dev/v1/scenes/chapel-q1b-1/occupancy.png \
         https://ply-api.ricco-ply.workers.dev/v1/scenes/chapel-q1b-1/topview.webp \
         https://ply-api.ricco-ply.workers.dev/v1/scenes/chapel-high-2/scene.json \
         https://ply-api.ricco-ply.workers.dev/v1/scenes/hq-classroom-1/scene.json \
         https://ply-api.ricco-ply.workers.dev/v1/scenes/lidar-office-8b/scene.json \
         https://github.com/Ricco-Mallick/ply \
         https://zenodo.org/records/10230733 \
         https://zenodo.org/records/10222321 ; do
  printf '%s  %s\n' "$(curl -s -o /dev/null -w '%{http_code}' -L --max-time 25 "$u")" "$u"
done
```

The four `walk.html?scene=...` links are the same check with the scene URL appended percent-encoded; they return 200 for
all four twins today, and they are the links a judge is most likely to click.
