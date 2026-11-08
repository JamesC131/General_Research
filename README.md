# M3: a native Soar model of self-control (draft v0)

Self-contained package (includes pyclarion-sc 0.5.0 for the world). Full explanation: `docs/Document3_Soar_models.pdf`, Part III.

## Requirements
As M1, plus `pip install soar-sml` (Soar 9.6.5).

## Commands
```bash
python -m unittest discover -s tests
python run_episode.py --variant native_soar --scenario capacity --condition busy --seed 5001 --trace
python run_episode.py --variant N1_no_chunking --scenario capacity --condition busy --seed 5001 --trace
python train_habit.py --show native_soar/habit_rl.soar    # the learned Soar-RL values
python analyze.py                                          # H4a-H4d
python verify.py
```

## Variants
`native_soar` (the model), `native_habit_only`, `native_always_deliberate` (baselines), `N1_no_chunking` (ablation).
CLARION's A1/A2 do not apply (M3 has no drives and no cost term).

## Files
| Path | Content |
|---|---|
| `native_soar/native_cleaner.soar` | the model: 45 Soar productions |
| `native_soar/habit_rl.soar` | the trained habit (41 Soar-RL rules) |
| `native_soar/native_cleaner.py` | bridge: discretized perception, reward, time budget, habit training |
| `native_soar/variants.py` | the model and its variants |
| `train_habit.py` | re-train or inspect the habit (keep a copy of habit_rl.soar first) |
# General_Research
This is the repository where all my current work is posted.
