# M3 native Soar: analysis of grid_seeds_1-200.csv

H4a: largest difference between the anticipation and constant conditions over 200 seeds and 6 measures = 0.0 -> identical

| Test | What | n seeds | Mean difference [CI] | p (Holm) | Supported |
|---|---|---|---|---|---|
| H4b | no shift (quiet - free < +0.05) | 200 | 0.000 [0.000, 0.000] | < .001 | yes |
| H4c | no capacity effect, chunking on (within +/-0.05) | 200 | 0.000 [0.000, 0.000] | < .001 | yes |
| H4d | capacity effect, no chunking (busy - free > 0) | 200 | 0.838 [0.810, 0.864] | < .001 | yes |

## Every variant

| Variant | Mean lapse | Room C % | Steps | Planning calls | Deliberations | Time-outs | H1b | H2 | H3 |
|---|---|---|---|---|---|---|---|---|---|
| N1_no_chunking | 0.136 | 15.2 | 196 | 59.4 | 59.4 | 11.7 | 0.000 | 0.000 | 0.838 |
| native_always_deliberate | 0.000 | 0.0 | 165 | 55.9 | 55.9 | 0.0 | 0.000 | 0.000 | 0.000 |
| native_habit_only | 1.000 | 100.0 | 300 | 0.0 | 0.0 | 0.0 | n/a | 0.000 | 0.000 |
| native_soar | 0.000 | 0.0 | 165 | 49.5 | 49.5 | 0.0 | 0.000 | 0.000 | 0.000 |

## Deliberations and time-outs per episode, capacity scenario

|                                      |   n_deliberations |   n_timeouts |   lapse_busy |
|:-------------------------------------|------------------:|-------------:|-------------:|
| ('N1_no_chunking', 'busy')           |            105.54 |        70.11 |         0.84 |
| ('N1_no_chunking', 'free')           |             46.48 |         0    |         0    |
| ('native_always_deliberate', 'busy') |             52.8  |         0    |         0    |
| ('native_always_deliberate', 'free') |             52.8  |         0    |         0    |
| ('native_habit_only', 'busy')        |              0    |         0    |         1    |
| ('native_habit_only', 'free')        |              0    |         0    |         1    |
| ('native_soar', 'busy')              |             46.48 |         0    |         0    |
| ('native_soar', 'free')              |             46.48 |         0    |         0    |
