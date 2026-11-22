# Cost analysis: a third-party GPU against our own

Round 2, 14 November 2026. Every price billed by the GPU provider is measured and appears in section 1 with the run it
belongs to. Everything on the own-GPU side is an assumption, listed in section 3, because no run of Ply has ever
executed on a card we own.

## 1. The measured runs

| Run | Steps | Wall clock | Billed | Implied rate | Cost per 1,000 steps |
|---|---:|---:|---:|---:|---:|
| Chapel training run | 15,000 | 38 min | $0.92 | $1.45/h | $0.061 |
| Longer training run | 30,000 | 36 min | $1.53 | $2.55/h | $0.051 |
| Chapel training run | 15,000 | 9 min | $0.38 | $2.53/h | $0.025 |
| Smoke run | 300 | 2.2 min | $0.087 | $2.37/h | $0.290 |
| Modal's list price for an L40S | | | $2.58/h | | |
| Metered on the experiments account this cycle | | | $15.26 | | |

The implied rate is the billed amount divided by the wall clock, and it is not the same across the four runs. Three of
them land at $2.37 to $2.55 an hour, just under the L40S list price. The 38-minute run lands at $1.45 an hour, so it ran
on a cheaper card: at the L4 rate of $1.43 an hour that `docs/EXPERIMENTS.md` records, 38 minutes bills $0.91, which is
the $0.92 measured. Treat "which GPU" as inferred from the arithmetic, not as a field in a report.

The same 15,000 steps cost $0.92 in 38 minutes on the slower card and $0.38 in 9 minutes on the faster one. Choosing the
faster GPU for that job was worth 2.4 times the price of the job, which is a larger effect than any ownership model in
this document produces at Ply's volume.

The four prices are whole-job bills as launched, so they include container start and whatever export ran with the
training. A full capture-to-twin job also pays the camera solve and the labelling pass, which the training runs above
reused from a saved solve. For scale, `docs/EXPERIMENTS.md` records the chapel's incremental solve at $0.73 on eight
CPU cores and the match pass that rebuilt its database at $0.31. Those two figures are outside the set of prices this
analysis was asked to use, and are quoted only so the per-run comparison is not read as a complete job.

## 2. What the numbers say before any hardware is bought

Three things, and they set up the rest of the document.

1. A run is minutes, not hours. At $2.53 an hour, a 9-minute run costs 38 cents and a 38-minute run costs 92.
2. The provider charges for wall clock, so the price of the same work is set mostly by which GPU is free when the job
   starts. Slow card, 2.4 times the bill.
3. Ply's measured spend is small. The experiments account was metered $15.26 this cycle, which at the measured prices is
   40 runs at $0.38 or 10 runs at $1.53.

## 3. Own GPU: the assumptions

| | Assumption | Value |
|---|---|---|
| A1 | Card | used RTX 4090, 24 GB, Rs 2,20,000 |
| A2 | Life | 5 years straight line, no resale value, no failure inside the life |
| A3 | Draw under load | 450 W, host included |
| A4 | Draw when idle | 60 W, machine left on |
| A5 | Electricity | Rs 8 per kWh |
| A6 | Exchange rate | Rs 88 = US$1 |
| A7 | Run profile | the 15,000-step chapel train plus export, 9 minutes, the wall clock the faster measured run took |
| A8 | Speed on the card | the same 9 minutes as that run, because no 4090 run exists. Section 6 repeats the answer at 1.5 times slower |
| A9 | Operator time | 0.25 h per run at Rs 500/h, so Rs 125 (about $1.42). Charged to both sides, so it cancels in the comparison |

The card has the same 24 GB as the L4 that trained the 2M-splat chapel model, so memory is not expected to be the
binding limit. That is an inference from the L4 runs, not a measurement on a 4090.

Fixed annual cost: Rs 2,20,000 / 5 = Rs 44,000. Electricity: 0.45 kW x Rs 8 = Rs 3.60 per loaded hour, 0.06 kW x Rs 8 =
Rs 0.48 per idle hour.

## 4. Per-run cost on the card

The card costs money whether it is working or not, so the honest per-run figure depends on how many runs a year it
gets. Write N for runs per year and U for utilisation, the share of the 8,760 calendar hours spent training:

    capital per run   = (Rs 2,20,000 / 5) / N
    idle electricity  = (8,760 h - N x 0.15 h) x Rs 0.48 / N
    load electricity  = 0.15 h x Rs 3.60 = Rs 0.54 per run
    total per run     = the three added, with N = U x 8,760 / 0.15

| Volume | Runs per year | Utilisation | Capital | Idle power | Load power | Total per run | Against $0.38 on demand |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ply's measured cycle, 30 runs a cycle, 4 cycles a year | 120 | 0.21% | Rs 366.67 | Rs 34.97 | Rs 0.54 | Rs 402.17 ($4.57) | 12.0x |
| Two cycles a month | 720 | 1.23% | Rs 61.11 | Rs 5.77 | Rs 0.54 | Rs 67.42 ($0.77) | 2.0x |
| Four runs a day | 1,460 | 2.5% | Rs 30.14 | Rs 2.81 | Rs 0.54 | Rs 33.48 ($0.38) | 1.0x |
| Eight runs a day | 2,920 | 5% | Rs 15.07 | Rs 1.37 | Rs 0.54 | Rs 16.98 ($0.19) | 0.51x |

At the volume Ply actually runs, the card is twelve times the price of the API per run. It only draws level at about
four runs a day, every day, for five years.

## 5. Break-even

Two ways to get the same answer, and the second is the useful one.

By hours. The card costs Rs 2,20,000 and then Rs 3.60 per loaded hour in electricity. The API costs Rs 227 an hour at
the L40S list price. Break-even is

    Rs 2,20,000 + 3.60 H = 227 H  ->  H = 985 loaded hours

which is 6,564 runs of 9 minutes, 1,641 runs of 36 minutes, or 11% of a year's 8,760 hours run flat out. Spread over the
card's five years that is 197 loaded hours a year, a utilisation of 2.25%.

By runs, with idle power counted: break-even is **1,462 runs a year** (about four a day, 2.5% utilisation). With the
machine switched off between runs it is 1,337 runs a year. Over the five-year life those are 6,685 to 7,310 runs.

Ply's measured volume is 10 to 40 runs a cycle. Even at 360 runs a year, break-even is four times away.

## 6. Sensitivity

Per-run cost on the card at the 9-minute profile, with the run count rounded to whole runs a year. A slower card is one
sentence, after the table.

| Utilisation | Runs per year | Capital | Idle power | Load power | Cost per run | Against $0.38 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.2% | 117 | Rs 376.07 | Rs 35.87 | Rs 0.54 | $4.69 | 12.33x |
| 0.5% | 292 | Rs 150.68 | Rs 14.33 | Rs 0.54 | $1.88 | 4.95x |
| 1% | 584 | Rs 75.34 | Rs 7.13 | Rs 0.54 | $0.94 | 2.48x |
| 2.5% | 1,460 | Rs 30.14 | Rs 2.81 | Rs 0.54 | $0.38 | 1.00x |
| 5% | 2,920 | Rs 15.07 | Rs 1.37 | Rs 0.54 | $0.19 | 0.51x |
| 10% | 5,840 | Rs 7.53 | Rs 0.65 | Rs 0.54 | $0.10 | 0.26x |
| 25% | 14,600 | Rs 3.01 | Rs 0.22 | Rs 0.54 | $0.04 | 0.11x |
| 50% | 29,200 | Rs 1.51 | Rs 0.07 | Rs 0.54 | $0.02 | 0.06x |

Utilisation is the whole argument: ten times the runs is a ten times cheaper run, and the hardware does not get cheaper
by being busy. A card half as fast again (13.5 minutes a run, assumption A8) moves the low rows by a few percent, since
the purchase price dominates them, and the 50% row by about half, where electricity stops being a rounding error.

## 7. With operator time included

At Rs 125 of operator time a run, charged to both sides:

| | On demand | Own card, 1% utilisation | Own card, 2.5% utilisation |
|---|---:|---:|---:|
| Compute | $0.38 | $0.94 | $0.38 |
| Operator | $1.42 | $1.42 | $1.42 |
| Total | $1.80 | $2.36 | $1.80 |

The operator line is the largest in every column and it changes no ranking: the same quarter hour is charged on both
sides, so it widens the numbers without moving the verdict. At 1% utilisation the card is 56 cents a run more than
renting; at four runs a day the two are level. Either way a run costs a couple of dollars, and the argument that decides
this is the capital, not the bill.

## 8. What Ply chose, and why

Ply runs every paid job on Modal, on demand, one job at a time, and has never owned a GPU. The reasons, in order of how
much they decided it:

1. **Volume.** The experiments account was metered $15.26 this cycle, which is 10 to 40 runs. Break-even on a Rs 2,20,000
   card is 1,337 to 1,462 runs a year. The gap is four times at the top of that range and twelve at the bottom.
2. **The GPU choice inside the API is the bigger lever.** The same 15,000-step job billed $0.92 on the slower card and
   $0.38 on the faster one. Picking the faster card when it is free saved more than owning hardware would have at any
   volume Ply reaches.
3. **Capital and risk.** Rs 2,20,000 spent up front is a bet that the project keeps needing 200 loaded hours a year for
   five years. The API turns that into $0.38 each time a job runs, which is the correct shape for a project that runs
   thirty jobs before a competition and then nothing for a month.
4. **Operations.** A local card needs drivers, CUDA builds for each architecture, a machine that stays up for a
   six-hour job, and a person awake when it fails. Modal rebuilds the image and restarts the job.
5. **No idle bill.** The assumed idle draw is 60 W, which is Rs 4,205 a year before the card does anything, plus the
   capital. Ply pays nothing between cycles.

What would change this answer: sustained volume above four runs a day, a workload that must not leave the building, or a
provider price above about $10 an hour, where a few hundred runs a year would break even. None of
those is true today, so the decision stands for Round 2 and for the on-site final.

## 9. What this analysis does not price

Storage on Backblaze B2 and Hugging Face, which is gigabytes per twin and billed monthly rather than per run. The
Cloudflare Worker, D1 and R2 for the queue and the API, which sit inside the free tier at this traffic. The capture
phone and the laptop. And the six-hour container cap on Modal, which has not been reached by any run in these numbers.
