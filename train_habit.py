"""
Re-train the Soar-RL habit of M3 (300 episodes, reward = dirt cleaned in any room, no deliberation).

    python train_habit.py                     # overwrites native_soar/habit_rl.soar (keep a copy!)
    python train_habit.py --out my_habit.soar --episodes 100
Then inspect it:  python train_habit.py --show native_soar/habit_rl.soar
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from native_soar.native_cleaner import HABIT_FILE, habit_table, train_habit  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--out", default=HABIT_FILE)
    ap.add_argument("--show")
    a = ap.parse_args()
    if a.show:
        for cond, v in sorted(habit_table(a.show), key=lambda r: (str(r[0]), -r[1])):
            print(f"{v:8.3f}  {cond.pop('name'):<12} {cond}")
    else:
        print("trained:", train_habit(episodes=a.episodes, out=a.out))
