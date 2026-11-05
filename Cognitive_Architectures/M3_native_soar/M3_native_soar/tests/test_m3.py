"""Tests of the M3 package (development seed 5001).   python -m unittest discover -s tests"""
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from native_soar import VARIANTS, run_one  # noqa: E402
from native_soar.native_cleaner import NativeSoarCleaner, habit_table  # noqa: E402
run = run_one
from pyclarion_sc.robot.controller import make_config                  # noqa: E402

DEV = 5001          # development seed: tests never use confirmatory seeds 1-200
TEMPTED = {"temptation": 0.9, "temptation_distance": 0.2, "in_assigned_with_dirt": False}


class TestNativeSoar(unittest.TestCase):
    def robot(self, **kw):
        return NativeSoarCleaner(make_config("anticipation", "anticipation"), seed=DEV, **kw)

    def test_habit_prefers_temptation(self):
        """The trained Soar-RL habit values the detour far above the task."""
        rows = habit_table()
        det = [v for c, v in rows if c["name"] == "detour" and c["temptation-level"] == "high"
               and c["battery-level"] == "high"]
        nxt = [v for c, v in rows if c["name"] == "next_room" and c["temptation-level"] == "high"
               and c["battery-level"] == "high"]
        self.assertTrue(det and nxt)
        self.assertGreater(min(det), max(nxt))

    def test_deliberation_controls_temptation(self):
        r = self.robot()
        p = dict(r.world.perceive(), **TEMPTED)
        d = r.decide(p, "approaching room C's door")
        r.close()
        self.assertEqual((d["option"], d["level"], d["deliberated"]), ("next_room", "top", True))

    def test_chunking_makes_control_cheaper(self):
        r = self.robot()
        p = dict(r.world.perceive(), **TEMPTED)
        first = r.decide(p, "approaching room C's door")["decisions_used"]
        second = r.decide(p, "approaching room C's door")["decisions_used"]
        chunks = r.agent.ExecuteCommandLine("print --chunks")
        r.close()
        self.assertLess(second, first)
        self.assertIn("chunk", chunks)

    def test_time_up_hands_control_to_habit(self):
        r = self.robot(chunking=False, budget=2)
        p = dict(r.world.perceive(), **TEMPTED)
        d = r.decide(p, "approaching room C's door")
        r.close()
        self.assertEqual((d["option"], d["level"]), ("detour", "bottom"))
        self.assertTrue(d["timed_out"])

    def test_habit_mode_never_deliberates(self):
        r = self.robot(mode="habit")
        p = dict(r.world.perceive(), **TEMPTED)
        d = r.decide(p, "approaching room C's door")
        r.close()
        self.assertFalse(d["deliberated"])
        self.assertEqual(d["option"], "detour")

    def test_no_goal_stack_growth(self):
        r = self.robot()
        r.run()
        depth = len(re.findall(r"==>S:", r.agent.ExecuteCommandLine("print --stack")))
        r.close()
        self.assertLessEqual(depth, 2)


class TestHarness(unittest.TestCase):
    def test_every_variant_runs(self):
        for v in VARIANTS:
            m = run(v, "shift", "quiet", DEV)
            self.assertIn("lapse_rate", m, v)
            self.assertGreater(m["steps"], 50, v)

    def test_baselines_bracket_lapses(self):
        self.assertEqual(run("native_always_deliberate", "anticipation", "anticipation", DEV)["lapse_rate"], 0.0)
        self.assertEqual(run("native_habit_only", "anticipation", "anticipation", DEV)["lapse_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
