# Comparing a Ply twin with another service

A comparison is only worth quoting if the two sides describe the same capture. The chapel comparison
was not one: the reference was a 28 m gothic church from a professional service, ours an 18 m chapel
from eight handheld clips. It still taught us the right lessons, but it is not a benchmark.

Rules for the next one:

1. **Same building, same walk.** If the other service cannot process our clips, re-shoot with the SOP and
   give both sides the same footage.
2. **Compare numbers first.** Frames per second of footage, poses registered, gaussians trained, gaussians
   shipped, file size, wall clock. Renders come after, because a viewer's tone mapping flatters anything.
3. **Fixed pose for images.** Same camera position, same field of view, same exposure setting in both
   viewers, or the picture is a comparison of tone curves.
4. **Record it here** with the date, the build (`report.json`), and the exact URLs, so nobody repeats it.

## Reference points

| Date | Run | Views | Poses | Gaussians in → out | SPZ | Wall clock |
|---|---|---|---|---|---|---|
| 2026-09 | aHolo classroom (their service) | 973 keyframes, 1024 px, 368 s (2.64/s) | not exposed | ~1.5 M | 14.5 MB | 32 min |
| 2026-10 | Ply `hq-classroom-1` | 542, 1600 px, 484 s (1.12/s) | 542/542, 0.82 px | 4.0 M → 1.35 M | 37.1 MB | 164 min |
| 2026-10 | Ply `hq-chapel-1` | 275, 515 s (0.53/s) | 271/275, 0.78 px | 4.0 M → 0.68 M | 20.7 MB | 50 min |

The two Ply rows were produced before the planning and gates landed; both would now fail the views check
and never reach a viewer.

## Dr Johnson (Deep Blending), 2026-11-02

A public photo set: 263 photos at 1332×876 of one house, from Kaggle `thnhdg/tandt-db`. It ran through `plypipe` on
Modal with `pipeline/modal/ply_bench.py`, which solves once and then trains each preset from that solve. The GPU was an
L40S, because Modal had no L4 free. The held-out set is every 30th view (9 views), scored with gsplat's PSNR/SSIM/LPIPS.

- **Solve:** SIFT, exhaustive pairs, global mapper. 260/263 views registered after 3 unanchored views were dropped.
  Reprojection error 0.69 px, 56k points, 6.2 min.
- **Align:** floor tilt 3.0°, 54 % floor inliers.
- **Labels:** 43 objects.

| Tier (`--preset`) | Steps | Gaussians, cap → exported | SPZ / lite | Train | Held-out PSNR / SSIM / LPIPS |
|---|---|---|---|---|---|
| low (`quick`) | 7,000 | 367k → 270k | 5.1 / 4.4 MB | 1.5 min | 29.02 / 0.891 / 0.274 |
| medium (`standard`) | 30,000 | 733k → 623k | 14.4 / 9.9 MB | 6.9 min | 29.63 / 0.912 / 0.213 |
| high (`high`) | 40,000 | 1.47M → 1.18M | 27.4 / 12.7 MB | 11.6 min | 29.13 / 0.910 / 0.207 |

- **Cost:** the whole benchmark used $2.35 of Modal credit: two 60-photo demo runs, the solve, and four trainings.
  The fourth training was the A/B below.
- **Reference:** the 3DGS paper reports 28.77 / 0.899 / 0.244 for Dr Johnson at 30k steps. It holds out every 8th
  view and uses the dataset's own poses, so treat the comparison as indicative only.
- **Tiers before the plan change:** quick, standard and high would all have trained 18,000 steps, with caps of
  400k / 579k / 579k. `standard` and `high` were the same run.
- **High vs medium:** high is perceptually sharper (lower LPIPS) but not higher PSNR on these 9 views. The 2× budget
  pays off in fine texture, not in average error.
- **A/B, MCMC noise scaled to the room (`--scale-mcmc`):** the quick preset scored 28.94 / 0.892 / 0.273 with it,
  against 29.02 / 0.891 / 0.274 without, at a median camera distance of 2.9 m. Export kept 88 % of the trained
  splats with the flag and 74 % without, but held-out quality is the same, so the flag stays off by default.
- **Results:** the twins, reports and held-out renders are in the HF results repo under `bench/drjohnson/{low,medium,high}/`.
