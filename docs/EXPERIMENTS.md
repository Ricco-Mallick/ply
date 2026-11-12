# Experiments

One row per run, whether it won or not, with the cost from Modal's billing (plan, rule 4). Tables below the first
section are printed by `scripts/eval_table.py` from each run's `metrics.json`.

## Free readings from saved runs (2026-11-11)

Read from the saved High chapel run (`ply-bench:chapel/high/job`, 60,000 steps, 2M cap) and its inputs. No run was
made.

| Question | Reading | Consequence |
|---|---|---|
| Which export filter removed 47% of the splats? | The opacity filter (`opacity > 0.01`): 2,000,000 in, 1,059,736 after it, 1,058,149 after the floater filter (1,587 removed). The crop was tried and rejected by the export's own scoring | The loss is decided in training, not in export |
| Why are they transparent? | 45% of the trained splats have opacity at or below 0.005 and 34% below 0.001. They are small (median largest axis 6 mm against 31 mm for the kept ones) and 75% lie inside the kept model's bounds. While relocation runs, 70,000 to 130,000 splats (3.5 to 6.5%) are found dead and moved every 100 steps. Relocation stops at step 49,700 (83% of the run); in the 10,300 steps after it, 940,000 splats fade out and nothing brings them back | The 2M cap is in use during training and half of it is lost in the last 17%. Raising the cap would add splats that fade the same way. Q5 (a weaker opacity regulariser) tests the cause directly; a later relocation stop is a second variant if Q5 helps |
| Step time | 35 to 36 steps a second at 2M splats; 1,718 s for 60,000 steps | 15,000 steps take about 7 minutes: the size of a phase 4 variant |
| Are the chapel clips HLG or log footage? | No. HEVC Main 10, `yuv420p10le`, transfer, primaries and matrix all `bt709`, limited range (DJI Osmo Action 5 Pro; one clip probed). The extracted frames have ordinary contrast and saturation (1st to 99th percentile of luminance 12 to 239) | No colour conversion is needed for the chapel. 0.11 records the tags for every future capture |
| How much does exposure vary? | Median frame luminance ranges from 27 to 148 (of 255) over 60 frames spread through the walk; up to 8% of a frame is clipped at the windows | This is what Q1 (exposure correction folded back) is for |
| Held-out score of the saved run | PSNR 25.59 dB, colour-corrected 26.03 dB, SSIM 0.872, LPIPS 0.212 on 41 views (every 30th) | The reference the 15k-step baselines sit under |

Not read: wall crossings on the chapel grid. That needs the walking evaluation (`eval/nav.py`), which is not built.

## Phase 1: baselines and noise floor (2026-11-11)

The chapel's saved solve (1,223 views), High settings, 15,000 steps, 2M cap, on an L40S (account opservemakesalt),
code `596c02d`. Two hold-outs: every 30th view (40 views), and a block of 20 consecutive frames (`000117` to `000136`,
clip `0051`) that no training view sits between. `q0a` and `q0b` are the same run twice.

| variant | steps | every cc-PSNR | every cc-SSIM | every cc-LPIPS | block cc-PSNR | block cc-SSIM | block cc-LPIPS | viewer PSNR desktop | viewer PSNR phone | splats kept | minutes | est. cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q0a | 15000 | 24.85 | 0.8382 | 0.2655 | 22.60 | 0.8052 | 0.1637 | 25.39 | 24.20 | 61% (1.22M) | 8.7 | $0.38 |
| q0b | 15000 | 24.71 | 0.8347 | 0.2666 | 22.05 | 0.8003 | 0.1667 | 25.02 | 23.93 | 61% (1.21M) | 8.8 | $0.38 |

| metric | baseline mean | noise floor | twice the floor |
|---|---:|---:|---:|
| every cc-PSNR | 24.78 | 0.14 | 0.28 |
| every cc-SSIM | 0.8364 | 0.0035 | 0.0070 |
| every cc-LPIPS | 0.2661 | 0.0011 | 0.0022 |
| block cc-PSNR | 22.32 | 0.54 | 1.09 |
| block cc-SSIM | 0.8028 | 0.0049 | 0.0098 |
| block cc-LPIPS | 0.1652 | 0.0030 | 0.0060 |
| viewer PSNR desktop | 25.21 | 0.37 | 0.74 |
| viewer PSNR phone | 24.07 | 0.27 | 0.55 |

- **The every-30th hold-out flatters the model by about 2.5 dB**: 24.78 dB there against 22.32 dB on the block.
- **What counts as a change.** PSNR has to move by 0.28 dB on the every-30th set (the plan's 0.2 dB is inside the
  noise) and by 1.1 dB on the block. LPIPS is the sharper instrument: the plan's 0.005 is more than twice its noise
  on the every-30th set (0.0022) and just under it on the block (0.0060).
- One pair gives one difference per metric, so the floor itself is a rough number. A third baseline would firm it up.

## Phase 4: first variants (2026-11-11)

Same solve, same settings, one change each.

| variant | change | every cc-PSNR | every cc-SSIM | every cc-LPIPS | block cc-PSNR | block cc-SSIM | block cc-LPIPS | viewer desktop | viewer phone | splats kept | see-through (worst view) | desktop file | est. cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline mean | | 24.78 | 0.8364 | 0.2661 | 22.32 | 0.8028 | 0.1652 | 25.21 | 24.07 | 61% (1.21M) | 0.02 to 0.14% (0.2 to 1.5%) | 38.9 MB | |
| q5a | opacity regulariser 0.005 | 24.97 | **0.8451** | **0.2466** | 22.56 | 0.8103 | **0.1567** | 25.24 | 24.28 | 72% (1.43M) | 1.8% (14.6%) | 44.8 MB | $0.34 |
| q5b | opacity regulariser 0.002 | **25.32** | **0.8483** | **0.2336** | 22.69 | **0.8208** | **0.1475** | 25.70 | 24.50 | 80% (1.60M) | 0.4% (3.0%) | 48.8 MB | $0.38 |
| q2 | pose refinement | 24.76 | 0.8339 | 0.2688 | 22.55 | 0.8097 | **0.1584** | 25.10 | 24.19 | 60% (1.20M) | 0.6% (5.6%) | 38.3 MB | $0.41 |
| q4a | sparse depth loss | diverged | | | | | | | | | | | $0.04 |

Bold is better than the baseline by more than twice the noise floor.

- **Q5, a weaker opacity regulariser: the first real gain.** At 0.002 every hold-out metric improves and all but
  block PSNR by more than twice the noise: +0.55 dB and −0.033 LPIPS (12%) on the every-30th set, −0.018 LPIPS on the
  block, and 80% of the trained splats survive export where 61% did. This is the mechanism the free readings
  pointed at. It is not free: the desktop file grows from 38.9 to 48.8 MB, and see-through pixels in the viewer rise
  from under 0.15% to 0.4% (3.0% in the worst view). 0.005 sits between the two on quality but shows more see-through
  than either (1.8%), which one run cannot explain.
  **Not the default yet.** Next: 0.001 and 0 to find where the trend ends, a look at the see-through views, then one
  full-length run of the winner before `train_launcher.py` changes.
- **Q2, pose refinement: no gain at this length.** Every-30th PSNR and SSIM are inside the noise and LPIPS is a hair worse (+0.0027, just outside it), the
  block's LPIPS is a little better (−0.0068), and training takes 4% longer. Not worth a default on this evidence.
- **Q4a, sparse depth loss: a defect, not a result.** The loss is NaN from about step 158 and the trainer aborts in
  the MCMC relocation near step 570. **Cause (found on a CPU, 2026-11-11):** gsplat's loss takes
  `where(depth > 0, 1 / depth, 0)`, and the gradient of that is NaN wherever the sampled depth is exactly 0. The
  bilinear sample hands the NaN to all four pixels around the track point, also to a pixel it weighs at exactly 0; if
  that pixel is rendered, every splat on it turns NaN. With gsplat's own function, 2,000 track points and a 1728 px
  frame this happens in 1 to 5 steps of 300 (by how much of the frame is rendered), so first somewhere between step
  60 and 300. The scene's scale is not the cause: disparity times scene scale is the same in metres as in a
  normalised scene. `train_launcher.py` now swaps in a loss with the same value and a finite gradient.
  **Confirmed on a GPU on Dr Johnson** (below): it trains to the end. Not run again on the chapel.

**Cost.** Modal's billing for the day on opservemakesalt: $3.17 ($2.88 for the eval runs, one cancelled export among
them, and $0.29 for the image and wheel checks). The runner's own estimates for the same runs add up to $2.32, so billing
runs about a quarter above them.

## Small scenario: Dr Johnson (2026-11-11, night)

From here on a variant is tried on a small twin first, and only a winner is repeated on the chapel (plan decision 8).
Dr Johnson's saved solve (260 photos at 1330×874, the `ply_bench` solve of 2026-11-02), High settings, 15,000 steps,
600k cap, 4 cores and 16 GiB, on an L4 or an L40S, whichever Modal had (account opservemakesalt;
`PLY_EVAL_SCENARIO=drjohnson`). Hold-outs: every 8th view (32 views) and a block of 10 photos in a row (`000128` to
`000137`). Code `7689f02` for the first four runs, `47dfbf2` (the export fix below) for the rest.

| variant | change | every cc-PSNR | every cc-SSIM | every cc-LPIPS | block cc-PSNR | block cc-SSIM | block cc-LPIPS | viewer desktop | viewer phone | splats kept | see-through (worst view) | desktop file | GPU | minutes | billed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| q0a | baseline | 29.26 | 0.8830 | 0.2410 | 27.55 | 0.8598 | 0.2669 | 29.37 | 28.79 | 74% (0.44M) | 0.12% (1.4%) | 12.5 MB | L40S | 3.8 | $0.19 |
| q0b | baseline | 29.24 | 0.8839 | 0.2414 | 27.60 | 0.8620 | 0.2640 | 29.38 | 28.77 | 74% (0.44M) | 0.41% (3.5%) | 12.4 MB | L4 | 8.5 | $0.20 |
| q4a | sparse depth loss | 29.18 | 0.8837 | 0.2399 | 27.62 | 0.8657 | 0.2628 | 29.37 | 28.74 | 73% (0.44M) | 0.51% (3.9%) | 12.6 MB | L40S | 5.0 | $0.26 |
| q5a | opacity regulariser 0.005 | 29.26 | **0.8860** | **0.2273** | **27.97** | **0.8686** | **0.2513** | **29.44** | **28.92** | 84% (0.51M) | 0.11% (0.8%) | 14.4 MB | L4 | 7.9 | $0.17 |
| q5b | opacity regulariser 0.002 | **29.35** | 0.8836 | **0.2181** | **27.90** | **0.8680** | **0.2485** | **29.47** | **28.95** | 90% (0.54M) | 0.12% (1.2%) | 15.2 MB | L40S | 4.0 + 1.5 | $0.20 + $0.11 |
| q5c | opacity regulariser 0.001 | 29.25 | 0.8817 | **0.2183** | 27.54 | 0.8622 | **0.2529** | **29.39** | **28.94** | 93% (0.56M) | 0.13% (1.1%) | 15.6 MB | L40S | 4.6 | $0.24 |
| q5d | no opacity regulariser | 28.59 | 0.8680 | **0.2331** | 27.37 | 0.8523 | 0.2653 | 28.62 | 28.22 | 96% (0.57M) | 0.07% (0.6%) | 16.0 MB | L40S | 4.5 | $0.24 |

Bold is better than the baseline mean by more than twice the noise floor. `q5b` was trained once; its export was run
a second time with the fix, and the viewer columns, splats kept and file size are from that second export.

| metric | baseline mean | noise floor | twice the floor |
|---|---:|---:|---:|
| every cc-PSNR | 29.25 | 0.02 | 0.04 |
| every cc-SSIM | 0.8835 | 0.0009 | 0.0018 |
| every cc-LPIPS | 0.2412 | 0.0004 | 0.0008 |
| block cc-PSNR | 27.58 | 0.05 | 0.10 |
| block cc-SSIM | 0.8609 | 0.0022 | 0.0044 |
| block cc-LPIPS | 0.2655 | 0.0029 | 0.0058 |
| viewer PSNR desktop | 29.37 | 0.01 | 0.01 |
| viewer PSNR phone | 28.78 | 0.02 | 0.03 |

- **The small scene is the sharper instrument, at about half the price.** Its noise is 0.02 dB and 0.0004 LPIPS
  on the every-8th set, against 0.14 dB and 0.0011 on the chapel. A run is billed $0.17 to $0.26; the chapel's were
  estimated at $0.34 to $0.43 and billed about a quarter more. The two baselines ran on different GPUs, so the floor
  includes that difference. It is still one pair.
- **Q5: 0.002 is where the trend ends, on a second scene.** LPIPS falls from 0.2412 to 0.2273 (0.005), 0.2181 (0.002)
  and 0.2183 (0.001), and rises again with no regulariser at all (0.2331, with PSNR down 0.67 dB). 0.002 is the only
  setting that is better or equal on every metric: +0.10 dB and −0.023 LPIPS (10%) on the every-8th set, +0.32 dB and
  −0.017 LPIPS on the block, +0.10 dB in the viewer and +0.17 dB on the phone file. It costs 22% in file size
  (15.2 against 12.5 MB; 12.2 against 10.1 MB for the phone).
- **Export was undoing that gain: a defect, fixed.** The first export of `q5b` shipped at 27.72 dB in the viewer with
  7.05% see-through pixels (15% in the worst view), from a trained model that scored 29.55 dB and 0.10% before
  cleanup. The value filter was chosen by splat count: `opacity>0.030 and scale<1.52` kept 88% of the splats, which
  passed the 85% rule, and it removed splats the picture needed. The three cleanups were measured and all failed
  the tolerance, and the lightest was shipped regardless. Export now loosens the filter when that happens
  (`choose_cleanup` in `stages/export.py`): the same model exports at 29.47 dB and 0.12% under `opacity>0.020` with
  no scale cap. Which of the two closed the holes, the lower floor or the missing cap, was not separated. The
  baselines and `q4a` take the same path as before. This is the likely cause of the chapel's
  see-through readings above (1.8% for `q5a`, 0.4% for `q5b`); their exports have not been run again.
- **Q4a: the depth loss trains now, and buys nothing here.** The run that went NaN on the chapel finished: the fix in
  `train_launcher.py` holds on a GPU. PSNR is 0.08 dB lower and LPIPS 0.0013 lower on the every-8th set, the block is
  inside the noise but for SSIM, and training takes 40% longer (237 against 169 s on an L40S). Not a default.
- **Not answered here:** pose refinement (not run on this scene), and whether any of this holds on the chapel.

### The opacity regulariser at twice the length

The same scene at 30,000 steps (ad-hoc variants: `--args "--iterations 30000 ..."`, `PLY_EVAL_TIMEOUT_MIN=27`), all
three on an L40S, code `47dfbf2`. One run each, so the 15k noise floor above is the only guide.

| variant | change | every cc-PSNR | every cc-SSIM | every cc-LPIPS | block cc-PSNR | block cc-SSIM | block cc-LPIPS | viewer desktop | viewer phone | splats kept | see-through (worst view) | desktop file | minutes | est. cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| q0-30k | baseline | 29.27 | 0.8881 | 0.2218 | 27.52 | 0.8656 | 0.2507 | 29.21 | 28.20 | 79% (0.47M) | 0.36% (2.6%) | 14.1 MB | 7.0 | $0.26 |
| q5a-30k | opacity regulariser 0.005 | 29.29 | 0.8880 | 0.2091 | 27.41 | 0.8599 | 0.2416 | 29.24 | 28.49 | 87% (0.52M) | 0.04% (0.2%) | 16.0 MB | 6.9 | $0.26 |
| q5b-30k | opacity regulariser 0.002 | 29.31 | 0.8840 | 0.2029 | 27.30 | 0.8474 | 0.2388 | 29.42 | 28.76 | 94% (0.56M) | 0.23% (1.5%) | 17.2 MB | 6.1 | $0.23 |

- **At length the regulariser is a dial between sharpness and fidelity, not a free gain.** 0.002 still lowers LPIPS
  by 0.019 (8.5%) on the every-8th set and by 0.012 on the block, and the viewer shows +0.21 dB (+0.56 dB on the phone
  file). But SSIM falls by 0.004 on the every-8th set and by 0.018 on the block, and the block loses 0.22 dB. The
  15k runs flattered it: with twice the steps the baseline catches up in PSNR and the SSIM gap opens.
- **0.005 keeps about two thirds of the gain for a third of the loss.** LPIPS −0.013 (5.7%) with PSNR and SSIM
  unchanged on the every-8th set; on the block LPIPS −0.009, PSNR −0.11 dB, SSIM −0.006. Its see-through pixels are
  the lowest of the three.
- **The default stays at 0.01.** The plan ships a variant only if no other metric gets worse by more than the noise,
  and both settings lose on the block by about twice the 15k floor. Whether that loss is real needs the floor at this
  length: a second 30k baseline (about $0.30) is the next run, before any chapel run.

**Cost.** Modal's billing for 2026-11-11 on opservemakesalt rose from $3.17 to $5.34: $2.17 for the small scenario
(the smoke run, seven variants, the second export of `q5b`, and most of two 30k runs). The last minutes of those two
and all of `q5a-30k` fall on 2026-11-12 and were not in the billing report yet: about $0.50 more. The runner's own
estimates for the eleven runs billed so far add up to $1.75: a launch pays $0.05 to $0.08 before its function starts,
which the estimate leaves out.

### The floor at 30k, and where the chapel's see-through pixels come from (2026-11-12)

Two runs to close the question above, both on an L40S at code `118b29e`: the 30k baseline a second time
(`q0-30k-b`, 6.4 minutes, billed $0.31) and the chapel's 0.005 twin exported again under the corrected filter rule
(`q5a-reexport`: the saved `q5a` training with only `export` run again, 2.5 minutes, billed $0.16).

**The floor at 30k on Dr Johnson**, and each setting against the mean of the two baselines:

| metric | q0-30k | q0-30k-b | floor | 0.005 (`q5a-30k`) | 0.002 (`q5b-30k`) |
|---|---:|---:|---:|---:|---:|
| every cc-PSNR | 29.27 | 29.36 | 0.09 | −0.02 | −0.01 |
| every cc-SSIM | 0.8881 | 0.8882 | 0.0001 | −0.00015 | −0.0042 |
| every cc-LPIPS | 0.2218 | 0.2198 | 0.0020 | −0.0117 | −0.0179 |
| block cc-PSNR | 27.52 | 27.42 | 0.11 | −0.06 | −0.17 |
| block cc-SSIM | 0.8656 | 0.8609 | 0.0047 | −0.0034 | −0.0159 |
| block cc-LPIPS | 0.2507 | 0.2515 | 0.0008 | −0.0095 | −0.0123 |
| viewer, desktop file | 29.21 | 29.29 | 0.08 | −0.01 | +0.17 |
| viewer, phone file | 28.20 | 28.36 | 0.16 | +0.21 | +0.48 |
| see-through (worst view) | 0.36% (2.6%) | 0.08% (0.9%) | 0.28 points | 0.04% (0.2%) | 0.23% (1.5%) |

- **0.005 clears the ship rule at 30k.** LPIPS falls by 0.012 on the every-8th set and by 0.010 on the block, several
  times the floor, and nothing else moves by more than the floor (SSIM on the every-8th set is half a unit of the last
  stored digit under the mean). The loss on the block that the table above showed (−0.11 dB, −0.006 SSIM) is the size
  of the difference between two identical baselines (0.11 dB, 0.0047). Its see-through pixels are the lowest of the
  four runs.
- **0.002 does not.** SSIM falls by 0.004 on the every-8th set and by 0.016 on the block, and the block loses
  0.17 dB: all beyond the floor. It is dropped.
- **See-through is the noisiest reading here.** Two identical baselines gave 0.36% and 0.08% (2.6% and 0.9% in the
  worst view). One run cannot place a setting on that scale.

**The chapel's 0.005 twin, exported again: the same file.** 1,430,582 splats, 25.24 dB in the viewer, 1.82% of its
pixels see-through and 14.6% in the worst view, as before. The corrected rule changes nothing because this export
did nothing wrong: the trained model itself, before any filter, already renders see-through
(`export.cleanupChoice.reference`, 12 held-out views, the splats covering less than 90% of a pixel).

| chapel run, 15k | opacity regulariser | trained model, see-through (worst view) | shipped file |
|---|---:|---:|---:|
| q0a | 0.01 | 0.02% (0.1%) | 0.02% (0.2%) |
| q0b | 0.01 | 0.13% (1.5%) | 0.14% (1.5%) |
| q5a | 0.005 | 1.76% (14.1%) | 1.82% (14.6%) |
| q5b | 0.002 | 0.37% (2.8%) | 0.40% (3.0%) |

On Dr Johnson the see-through pixels of the 0.002 twin were the export's doing and the fix removed them. On the
chapel they are training's: with the weaker regulariser the model has gaps that the stronger one closes, 13 times the
difference between the two baselines at 0.005 and 3 times at 0.002. One run each, so the order of the two says
nothing. A likely cause, not tested: the regulariser is what feeds MCMC's relocation (a splat pushed under the
opacity threshold is moved to where the model is dense), so a weaker one moves fewer splats onto surfaces that began
with few, and 15k steps on a 2M-splat room are not enough to cover them otherwise. Dr Johnson at 30k, where 0.005
has the fewest see-through pixels, fits that.

**Decision (plan, decision 10): the default stays at 0.01.** 0.005 is a gain on the small scene at both lengths, but
a default has to hold on the demo scene, and export cannot repair a hole that training left. Another 15k chapel run
would not settle it: a clean repeat would still leave one run in two with a hole. A full-length chapel run would, and
decision 8 puts that after the rest of the plan, so 0.005 goes to the combined run of phase 7 as a candidate, gated
on see-through no higher than the baseline's. The 3M cap is not run: it was conditional on 85% of the splats
surviving export, and the chapel keeps 72% at 0.005 and 80% at 0.002.

**Cost.** opservemakesalt's billing for 2026-11-12 rose from $1.59 to $2.06 with these two runs. Two launches that
went wrong beside them billed nothing (`MISTAKES.md`, 2026-11-12).

### Q6: the phone file, trained on at its own size (2026-11-12)

One run, `q6` on Dr Johnson (L40S, code `c055ad7`): the baseline's training, then an export with `--lite-max 240000
--lite-finetune 3000`. The twin ships 441k splats, so its phone file is cut to 240k (the chapel's 2M to 800k ratio),
and that model is then trained on for 3,000 steps with the splat cap at its own count (`plypipe/finetune.py`), on the
same training views. The export scores both phone models as the viewer shows them, on the 12 held-out views, and
ships the better one.

| phone model | splats | viewer PSNR | SSIM | see-through (worst view) | below the desktop file |
|---|---:|---:|---:|---:|---:|
| cut down, as it shipped until now | 240k | 28.36 | 0.8824 | 0.01% (0.13%) | 0.96 dB |
| cut down, then 3,000 steps | 240k | 29.09 | 0.8913 | 0.05% (0.41%) | 0.23 dB |
| the same as shipped (`scene_lite.spz`, 5.6 MB) | 240k | 29.05 | 0.8912 | 0.04% (0.39%) | 0.22 dB |
| for scale: `q0a`'s phone file, never cut (10.1 MB) | 443k | 28.79 | | | 0.59 dB |

The desktop file of this run: 29.32 dB cleaned, 29.27 dB as shipped (12.5 MB). The two baselines' phone files differ
by 0.02 dB, so the noise is far below these differences.

- **+0.73 dB at the same 240k splats: three quarters of the gap to the desktop file is gone.** The plan ships Q6 when
  the phone file comes within 0.7 dB of the desktop file at the same size; it is at 0.22 dB.
- **It also beats the phone file the baselines ship, at 54% of the splats and 56% of the bytes** (29.05 dB against
  28.79 dB). Those files were never cut down, only stripped of their third SH band, and that alone costs 0.59 dB: a
  model trained with the bands it ships gets it back. In this run the export fine-tuned only a phone file it had cut
  down; a twin under the phone size is covered since `60db587` and measured below.
- **Away from the training views the phone model holds up.** The trainer's own scores for it: cc-PSNR 29.14 on the
  every-8th set (desktop 29.26) and 28.03 on the block (desktop 27.70); cc-LPIPS 0.248 and 0.267 (desktop 0.241 and
  0.265).
- **See-through pixels rise from 0.01% to 0.05%** (0.41% in the worst view), inside what the export tolerates (one
  point more than the cut-down model).
- **Time:** 67 s of the export for 3,000 steps at 240k splats on an L40S, the trainer's start and both scorings
  included; the whole export took 123 s.
- **Off by default** (`--lite-finetune STEPS`). The chapel's repeat and the time it adds are below.

**Cost.** Billed $0.28 on opservemakesalt for the run (estimated $0.24 on an L4 and $0.34 on an L40S; the L40S did it
in 5.5 minutes).

#### The chapel, and a twin that is not cut down

Two more runs on an L40S, one launch after the other. `q6` on the chapel (code `9713f9a`): 1.21M splats shipped, the
phone file cut to 800k. `q6-uncut` on Dr Johnson (`--args "--lite-finetune 3000"`, code `60db587`): from that commit
the fine-tune also covers a twin under the phone size, whose phone file keeps its 441k splats and loses only the third
SH band.

| run | phone model | splats | viewer PSNR | SSIM | see-through (worst view) | below the desktop file |
|---|---|---:|---:|---:|---:|---:|
| chapel `q6` | cut down | 800k | 23.89 | 0.8595 | 0.16% (1.8%) | 1.25 dB |
| | then 3,000 steps | 800k | 24.69 | 0.8655 | 0.29% (2.7%) | 0.45 dB |
| | the same as shipped (20.5 MB) | 800k | 24.64 | 0.8654 | 0.29% (2.7%) | 0.48 dB |
| Dr Johnson `q6-uncut` | third SH band cut | 441k | 28.70 | 0.8913 | 0.30% (1.7%) | 0.59 dB |
| | then 3,000 steps | 441k | 29.22 | 0.8953 | 0.30% (1.6%) | 0.07 dB |
| | the same as shipped (10.3 MB) | 441k | 29.19 | 0.8952 | 0.31% (1.7%) | 0.05 dB |

Desktop files: the chapel 25.14 dB cleaned and 25.12 dB as shipped (38.8 MB); Dr Johnson 29.29 and 29.24 dB (12.4 MB).

- **The chapel repeats it: +0.80 dB at the same 800k splats**, nearly six times its noise floor (0.14 dB). The phone
  file ships 0.48 dB below the desktop file, inside the plan's 0.7 dB. The two chapel baselines' phone files, cut to
  800k and left at that, ship 1.19 and 1.09 dB below theirs.
- **A twin that is not cut gains 0.52 dB**, and its phone file ends 0.05 dB below the desktop file: the SH band it
  gives up now costs next to nothing.
- **Away from the training views the phone models are not the weaker ones.** On the block of consecutive frames
  they score above their desktop models: 23.18 against 22.52 dB cc-PSNR on the chapel (cc-LPIPS 0.164 against 0.165),
  27.83 against 27.57 dB on Dr Johnson (0.256 against 0.266). On the every-Nth views they are 0.22 and 0.08 dB below.
- **See-through pixels are the one number that moves the wrong way:** 0.16% to 0.29% on the chapel, 1.8% to 2.7% in
  its worst view. That is inside what the export tolerates, and unchanged on the twin that is not cut.
- **Time:** 151 s of the chapel's export for 3,000 steps at 800k splats on an L40S (the export took 265 s against
  114 and 117 s for the baselines); 65 s on Dr Johnson at 441k. An L4 is about twice as slow, so about five minutes a
  job on the worker, inside the export stage's 40-minute limit (exports take 2 to 18 minutes today). At the L4's
  $1.43 an hour that is about $0.12 a job.
- **Off by default in these runs.** The plan's test is passed on both scenes; whether the gain is worth its price on
  every job is weighed below.

**Cost.** Billed on opservemakesalt on 2026-11-12: $0.58 for the chapel run (estimated $0.60) and $0.24 for
`q6-uncut` (estimated $0.34 on an L40S). With the first `q6` run, $1.11 for the three.

#### What it buys for its price: on by default (2026-11-12)

The owner's condition for turning it on: a significant gain for what it costs. Every lever measured in this file or
in `BENCHMARK.md` that buys image quality with GPU time, on an L40S at the run's own container price ($2.58 an hour
for the chapel, $2.27 for Dr Johnson):

| lever | gain in PSNR | GPU time | cost | dB per dollar |
|---|---:|---:|---:|---:|
| phone file fine-tuned, chapel (cut to 800k) | +0.80 dB, phone file | 151 s | $0.11 | 7 |
| phone file fine-tuned, Dr Johnson (cut to 240k) | +0.73 dB, phone file | 67 s | $0.04 | 17 |
| phone file fine-tuned, Dr Johnson (not cut) | +0.52 dB, phone file | 65 s | $0.04 | 13 |
| Quick to Standard training, Dr Johnson (`BENCHMARK.md`) | +0.61 dB, both files | 5.4 min | $0.20 | 3 |
| Standard to High training, Dr Johnson (`BENCHMARK.md`) | −0.50 dB | 4.7 min | $0.18 | none |
| 15k to 30k steps, Dr Johnson (above) | −0.16 dB in the viewer (cc-LPIPS −0.019) | 2 min | $0.06 | none |

- **Per dollar it buys two to six times what the next best lever buys**, and that lever (Standard instead of Quick)
  is the only other one with a gain in PSNR at all: past Standard, more training time bought none.
- **It is a small part of a job.** On the worker's L4 (about twice as slow as the L40S, at $1.43 an hour; the L4 is
  not timed) the chapel's fine-tune is about 5 minutes and $0.12, on a High job of about 135 GPU-minutes and $3.20:
  under 4% of either. A small twin pays about 2 minutes and $0.05.
- **The gain is far outside the noise:** 2.9 times the plan's bar on the chapel (0.28 dB, twice its noise floor) and
  2.6 to 3.7 times the 0.2 dB bar on Dr Johnson.
- **It reaches most viewers of a phone product:** the viewer loads the phone file on every phone and on low-tier
  desktops (`app/src/walk/scene.ts`).
- **What it costs in quality:** see-through pixels on the chapel, 0.16% to 0.29%.

So the export fine-tunes the phone file by default (`plan.LITE_FINETUNE_STEPS`, 3,000 steps; `--lite-finetune 0`
turns it off), with two limits that keep it to where it was measured to pay:

- **Only where the phone file gives something up:** fewer splats than the twin (cut to 800k) or fewer SH bands
  (Standard and High train SH 3, the phone file ships SH 2). A Quick twin (SH 2, under 800k) is its own phone file:
  nothing was lost, nothing is trained, and the job pays nothing.
- **Only where it can be scored:** a GPU and held-out views. Without them the report says so and the file ships as it
  was; with them the fine-tuned file ships only when the viewer shows it better.

The eval runner counts the 3,000 steps in every estimate from here on, and new runs' phone-file numbers are not
comparable with the ones above the Q6 heading (`--args "--lite-finetune 0"` gives the old export).

**On the live worker.** Deployed as v19 (`c5cccb3`) and run through the API with the smoke clip (job `8582285e…`;
the smoke job now trains at SH 3 so that its phone file gives up a band). The fine-tune ran inside the export in 44 s
(3,000 steps on 22k splats; the export took 62 s against 15 s for the morning's smoke job) and was not shipped: the
fine-tuned file scored 19.21 dB against 17.69 dB, but with 5.6% of its pixels see-through against 3.9%, more than
the one point the export allows. A 2,000-step model of a 30-second clip is not a twin (the gate fails it as `tiny`),
so this shows the path and the guard on the worker, not the gain. The GPU type the job got is not in its report.
Not yet seen on the worker: a phone file of 800k splats on an L4, in time and in memory.
