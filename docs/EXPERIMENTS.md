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
