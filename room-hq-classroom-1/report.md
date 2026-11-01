# Measurement sheet

- floor at y = 0.00 m
- ceiling estimate 7.19 m (splat percentile: verify against frames)

## Walls (5)

| # | from | to | length (m) | angle (deg) |
|---|---|---|---|---|
| 0 | (5.48, 5.25) | (-5.35, 5.45) | 10.83 | 179.0 |
| 1 | (-5.89, 4.55) | (-5.87, -5.54) | 10.09 | -89.9 |
| 2 | (5.64, -5.06) | (5.67, 4.99) | 10.04 | 89.8 |
| 3 | (-5.30, -5.66) | (4.63, -5.66) | 9.93 | 0.0 |
| 4 | (4.63, -5.66) | (5.64, -5.06) | 1.17 | 30.6 |

## Openings (1)

| kind | wall | at (m along) | width | sill | height |
|---|---|---|---|---|---|
| door | 2 | 2.84 | 0.10 | 0.88 | 0.86 |

## Objects (194)

| kind | count |
|---|---|
| chair | 87 |
| table | 45 |
| desk | 27 |
| ceiling fan | 10 |
| window | 6 |
| door | 5 |
| blackboard | 4 |
| cabinet | 3 |
| podium | 2 |
| shelf | 2 |
| projector screen | 1 |
| sign | 1 |
| stained glass window | 1 |

### The 30 largest (model these properly)

| id | kind | centre (x, z) | footprint (m) | height (m) |
|---|---|---|---|---|
| table-27 | table | (0.58, -1.82) | 1.04 × 1.17 | 0.13 |
| desk-18 | desk | (-0.02, -1.59) | 0.96 × 1.27 | 0.25 |
| table-42 | table | (-0.83, 1.96) | 1.17 × 0.97 | 0.22 |
| chair-53 | chair | (4.57, 0.70) | 1.10 × 0.69 | 0.12 |
| table-31 | table | (-0.01, -1.56) | 0.70 × 1.01 | 0.10 |
| chair-44 | chair | (-0.36, -1.58) | 0.96 × 0.67 | 0.59 |
| table-25 | table | (0.74, -0.89) | 0.68 × 0.93 | 0.17 |
| desk-27 | desk | (1.95, -1.04) | 0.38 × 1.64 | 0.10 |
| table-35 | table | (-0.20, 0.84) | 0.60 × 1.01 | 0.10 |
| table-39 | table | (0.45, 0.34) | 1.05 × 0.57 | 0.19 |
| desk-1 | desk | (-4.15, 0.27) | 0.40 × 1.47 | 0.10 |
| table-22 | table | (0.87, -0.17) | 0.51 × 1.13 | 0.16 |
| chair-37 | chair | (4.92, 1.19) | 0.62 × 0.83 | 0.31 |
| table-21 | table | (0.83, 0.71) | 0.53 × 0.92 | 0.15 |
| desk-7 | desk | (-0.65, -0.72) | 0.46 × 1.07 | 0.57 |
| desk-19 | desk | (-0.69, -0.74) | 0.44 × 1.09 | 0.57 |
| table-1 | table | (-3.97, 0.31) | 0.25 × 1.92 | 0.61 |
| chair-43 | chair | (3.80, -2.11) | 0.70 × 0.65 | 0.22 |
| desk-6 | desk | (0.62, 1.36) | 0.53 × 0.81 | 0.10 |
| chair-46 | chair | (1.16, 0.79) | 0.62 × 0.70 | 0.25 |
| chair-38 | chair | (-0.12, 1.84) | 0.79 × 0.54 | 0.16 |
| table-11 | table | (0.94, -1.61) | 0.54 × 0.78 | 0.13 |
| desk-25 | desk | (0.45, 0.99) | 0.65 × 0.66 | 0.21 |
| chair-26 | chair | (1.22, 1.31) | 0.60 × 0.70 | 0.24 |
| desk-21 | desk | (-0.56, 0.75) | 0.82 × 0.49 | 0.10 |
| table-17 | table | (2.20, 1.20) | 0.51 × 0.79 | 0.10 |
| chair-78 | chair | (4.06, -2.27) | 0.65 × 0.59 | 0.13 |
| table-3 | table | (2.18, 1.67) | 0.39 × 0.97 | 0.10 |
| table-16 | table | (1.85, -0.75) | 0.36 × 1.04 | 0.10 |
| chair-71 | chair | (-0.26, -1.49) | 0.82 × 0.45 | 0.10 |

- spawn point: (-2.61, 1.60, 2.65)

## Rules

- metres, +Y up, floor at 0, scene frame
- measurements are ±5 cm from the photogrammetric solve; say so where it matters
- model the hero objects, proxy the rest
- verify with scripts/render_pose.mjs + scripts/compare.py