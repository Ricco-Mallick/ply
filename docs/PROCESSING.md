# Processing settings: measured, not fixed

Every capture is different. A 30 s scan of a store room and an 8 min walk of a floor need different
frame rates, different working resolutions and different gaussian budgets. `plypipe` therefore plans
each run from the input itself; there is no house number to remember, and no preset to copy from
another product.

The plan is written to `<out>/plan.json` before the stage that uses it, and summarised in `report.json`,
so a finished job can always be audited: what was decided, from what measurement, and why.

| Measured | Sets | Rule |
|---|---|---|
| Clip duration and count (ffprobe) | kept frames per second | one frame per ~0.46 s of walking, scaled by how fast the operator actually moved, capped by a frame budget of `2 × seconds + 400` (700–3200 frames) |
| Motion samples from `capture.json` (flow %/s, turn °/s, 80th percentile) | the same rate | slow walk 0.7×, normal 1.0×, fast 1.6× |
| Source pixels (long side) | working resolution | 45 % of the source, snapped to 16 px, kept within 1280–2048 |
| Solved bounding box + shell ceiling | gaussian cap (MCMC) | ~4,200 gaussians per m² of wall + floor + ceiling surface, at most ~2,200 per solved view, 0.4M–4M |
| Views that registered | training steps | ~50 steps per view, 18k–55k, scaled by the preset |
| The trained PLY itself | export thresholds | opacity and scale filters are probed against the model; a filter that would keep less than 85 % of what training fitted is loosened until it does |

The three presets (`quick`, `standard`, `high`) are multipliers on that plan, not separate settings:
`quick` trains ~35 % of the steps with half the gaussians, `high` trains ~25 % more with 15 % more
gaussians. Explicit flags (`--fps`, `--long-side`, `--iterations`, `--cap-max`, `--min-opacity`) still
win over the plan when someone needs to force something for an experiment.

## Why these rules

Two runs of the same classroom are the reference point. A commercial service (aHolo) delivered a
visibly sharper twin from the same room with **973 views at 1024 px, 692k gaussians, 32 minutes**.
Ply's run used **542 views at 1600 px, 4M trained down to 1.35M at export, 3.4 hours** and looked
softer. More pixels, more gaussians and six times the compute produced a worse result.

What that ruled out and ruled in:

- **Not resolution.** 1024 px keyframes beat 1600 px frames. Pixels are cheap to add and do not
  rescue a soft fit.
- **Not gaussian count.** 692k beat 1.35M. Spending budget on gaussians that export later deletes is
  the same as not spending it.
- **Views per metre.** 2.6 poses/s against 1.1/s. Each gaussian has to cover less surface when the
  room was photographed more often.
- **Fitting, not filtering.** Training to a 4M cap and deleting 66 % at export leaves the survivors,
  usually long and translucent. Training to the size that will ship lets MCMC relocate instead of
  accumulate.

## Checklist after a real run

Read `report.json` (or `/v1/jobs/<id>`), not the UI:

- `frames.kept / frames.clips[].durationSec` at or above `quality.metrics.viewsFloor` (80 % of the planned rate), and no
  `frame cap` warning in the log.
- `train.antialiased` is `true`.
- `export.gaussiansOut / export.gaussiansIn` ≥ 0.85, and `exportFilters.why` is not the last-resort
  opacity-only branch.
- `plan.model.why` matches the room (surface area within a couple of m² of a tape measure).

## The gate: what a run must pass before anyone sees it

`plypipe/verify.py` reads `report.json` at the end of every run, on the GPU worker and locally, and the
worker refuses to publish a twin that fails it. The thresholds are the ones the chapel broke:

| Check | Fail | Warn |
|---|---|---|
| views per second of footage | below 80 % of the planned rate, within 1.2–1.8 (1.8 without `plan.json`) | below 90 % of it (2.1 without a plan) |
| floater pass recorded (`export.floaterClean`) | missing | — |
| share of trained splats kept by export | below 0.5 | below 0.85 |
| twin size | below 3 MB | — |
| registered frames | — | below 90% |
| antialiasing | — | off |
| splats at the top of the colour range | — | above 2% |

A failed run is stopped before upload, so the status page says why instead of linking to a twin nobody
would accept. Run it by hand on any finished job:

```bash
python -m plypipe.verify work/my-room/report.json
```

Two of those checks exist because a stage used to skip itself silently:

- **Floaters** are removed by `plypipe/clean.py`, a numpy pass over the exported PLY. The old `-F`
  filter needed WebGPU, and a container without an adapter skipped it without failing. The numpy pass
  either runs or the stage errors.
- **Colour** is not corrected at export, because a per-splat tweak there changes hue per view. The
  viewer carries a display gain instead (`[` dimmer, `]` brighter, `\` reset, persisted per browser),
  which is where a tone curve belongs.

## Don'ts

- Don't pass `--max-frames` to make a run faster. It silently lowers the kept rate (the classroom ran
  at 1.1 frames/s because of one) and no amount of training recovers the missing views. The stage now
  logs a warning when a cap lowers the rate.
- Don't raise `--cap-max` for a single room. Past the plan's cap the extra gaussians compete for the
  same views and turn into floaters.
- Don't trust an export filter that was never measured. `PLY_EXPORT_PROBES=0` made a 4M model export
  as 1.35M with no check; probes are on by default.
- Don't compare renders at different exposures and call it quality. Compare a fixed pose, or compare
  `report.json`.
- Don't edit a finished splat by hand. Every hand edit so far produced a worse file than the pipeline
  did: colour changes that ignore the higher-order SH shift hue per view. If a post-process is needed,
  it goes in `plypipe/` with a test, so the next run benefits too.
