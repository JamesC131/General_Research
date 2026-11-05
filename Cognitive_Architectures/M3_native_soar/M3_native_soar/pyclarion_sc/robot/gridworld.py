"""
A grid world for a Roomba-style cleaning robot.

Layout (default):

    ###########################
    #......#...........#......#
    #......#...........#......#
    #..A...#.....C.....#...B..#     A, B: rooms the robot is assigned to clean
    #......#...........#......#     C:    a room it is NOT assigned to (very dirty)
    #......#...........#......#
    ###.#######.#########.#####     doors to A, C and B
    #.........................#     corridor
    #D........................#     D: charging dock
    ###########################

The robot moves one cell per step, cleans the cell it is on, and uses battery.
It sees dirt in its own room and, through an open door, the room beyond it.
This module knows nothing about Clarion: it only executes behaviours
("options") chosen by a controller and reports what the robot perceives.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

LAYOUT = [
    "###########################",
    "#......#...........#......#",
    "#......#...........#......#",
    "#......#...........#......#",
    "#......#...........#......#",
    "#......#...........#......#",
    "###.#######.#########.#####",
    "#.........................#",
    "#D........................#",
    "###########################",
]

ROOMS = {  # name -> (row0, row1, col0, col1) inclusive interior
    "A": (1, 5, 1, 6),
    "C": (1, 5, 8, 18),
    "B": (1, 5, 20, 25),
    "corridor": (7, 8, 1, 25),
}
DOORS = {"A": (6, 3), "C": (6, 11), "B": (6, 21)}
DOCK = (8, 1)

OPTIONS = ["clean_quiet", "clean_full", "next_room", "detour", "dock"]
OPTION_TEXT = {
    "clean_quiet": "clean here (low suction)",
    "clean_full": "clean here (full suction)",
    "next_room": "go to the next assigned room",
    "detour": "detour to visible dirt",
    "dock": "return to the dock",
}


@dataclass
class WorldConfig:
    assigned: tuple = ("A", "B")
    tempting: str = "C"
    dirt_assigned: tuple = (0.25, 0.75)      # dirt range in assigned rooms
    dirt_tempting: tuple = (0.75, 1.0)       # the unassigned room is very dirty
    dirt_fraction: float = 0.7               # share of cells that start dirty
    quiet_zone: str | None = None            # room where low suction is required (scenario B)
    battery_move: float = 0.0025
    battery_clean: float = 0.0015
    battery_plan: float = 0.001              # extra cost of one explicit (top-level) decision
    charge_rate: float = 0.05
    sight_radius: float = 4.5                # how far through a door the robot can see
    clean_threshold: float = 0.05


@dataclass
class GridWorld:
    cfg: WorldConfig = field(default_factory=WorldConfig)
    seed: int = 0

    def __post_init__(self):
        self.walls = np.array([[c == "#" for c in row] for row in LAYOUT])
        self.H, self.W = self.walls.shape
        self.room_of = np.full((self.H, self.W), "", dtype=object)
        for name, (r0, r1, c0, c1) in ROOMS.items():
            self.room_of[r0:r1 + 1, c0:c1 + 1] = name
        for name, rc in DOORS.items():
            self.room_of[rc] = "door"
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self):
        rng = np.random.default_rng(self.seed)
        self.dirt = np.zeros((self.H, self.W))
        for name in self.cfg.assigned + (self.cfg.tempting,):
            r0, r1, c0, c1 = ROOMS[name]
            lo, hi = self.cfg.dirt_tempting if name == self.cfg.tempting else self.cfg.dirt_assigned
            block = rng.uniform(lo, hi, (r1 - r0 + 1, c1 - c0 + 1))
            mask = rng.random(block.shape) < (1.0 if name == self.cfg.tempting else self.cfg.dirt_fraction)
            self.dirt[r0:r1 + 1, c0:c1 + 1] = block * mask
        self.initial_dirt = self.dirt.copy()
        self.pos = DOCK
        self.battery = 1.0
        self.t = 0
        self.planning_calls = 0
        self.cleaned = {"assigned": 0.0, "unassigned": 0.0}
        self.collided = 0
        return self.perceive()

    # ---------------------------------------------------------------- helpers
    def passable(self, rc):
        r, c = rc
        return 0 <= r < self.H and 0 <= c < self.W and not self.walls[r, c]

    def neighbours(self, rc):
        r, c = rc
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if self.passable(n):
                yield n

    def path(self, start, goals):
        """Shortest path (BFS) from start to any cell in goals. Returns the list of cells or None."""
        goals = set(goals)
        if not goals:
            return None
        if start in goals:
            return [start]
        prev = {start: None}
        q = deque([start])
        while q:
            cur = q.popleft()
            for n in self.neighbours(cur):
                if n in prev:
                    continue
                prev[n] = cur
                if n in goals:
                    out = [n]
                    while prev[out[-1]] is not None:
                        out.append(prev[out[-1]])
                    return out[::-1]
                q.append(n)
        return None

    def room(self, rc=None):
        return self.room_of[rc or self.pos]

    def cells(self, name, dirty=True):
        r0, r1, c0, c1 = ROOMS[name]
        return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                if not dirty or self.dirt[r, c] > self.cfg.clean_threshold]

    def assigned_remaining(self):
        """Share of the assigned dirt still left, counting only cells the robot treats as dirty."""
        def room_sum(m, n, thr):
            r0, r1, c0, c1 = ROOMS[n]
            block = m[r0:r1 + 1, c0:c1 + 1]
            return float(block[block > thr].sum())
        tot = sum(room_sum(self.initial_dirt, n, 0.0) for n in self.cfg.assigned)
        left = sum(room_sum(self.dirt, n, self.cfg.clean_threshold) for n in self.cfg.assigned)
        return left / tot if tot > 0 else 0.0

    def visible_unassigned(self):
        """Dirty unassigned cells the robot can see: inside that room, or through its door."""
        name = self.cfg.tempting
        door = DOORS[name]
        here = self.room()
        cells = self.cells(name)
        if here == name:
            return cells
        if np.hypot(self.pos[0] - door[0], self.pos[1] - door[1]) <= self.cfg.sight_radius:
            return [rc for rc in cells if np.hypot(rc[0] - door[0], rc[1] - door[1]) <= self.cfg.sight_radius + 1]
        return []

    def next_assigned_room(self):
        for name in self.cfg.assigned:
            if self.cells(name):
                return name
        return None

    # -------------------------------------------------------------- perception
    def perceive(self) -> dict:
        here = self.room()
        assigned_here = here in self.cfg.assigned and bool(self.cells(here))
        vis = self.visible_unassigned()
        temptation, dist = 0.0, 1.0
        if vis:
            best = max(vis, key=lambda rc: self.dirt[rc])
            temptation = float(self.dirt[best])
            p = self.path(self.pos, [best])
            dist = min(1.0, (len(p) - 1) / 20.0) if p else 1.0
        remaining = self.assigned_remaining()
        return {
            "t": self.t, "pos": self.pos, "room": here, "battery": self.battery,
            "in_assigned_with_dirt": assigned_here,
            "quiet_zone": self.cfg.quiet_zone is not None and here == self.cfg.quiet_zone,
            "temptation": temptation, "temptation_distance": dist,
            "assigned_remaining": remaining, "assigned_done": remaining == 0.0,
            "at_dock": self.pos == DOCK,
            "next_room": self.next_assigned_room(),
        }

    # ---------------------------------------------------------------- dynamics
    def _clean(self, amount_factor):
        r, c = self.pos
        before = self.dirt[r, c]
        self.dirt[r, c] = before * (1 - amount_factor)
        got = before - self.dirt[r, c]
        kind = "assigned" if self.room() in self.cfg.assigned else "unassigned"
        self.cleaned[kind] += got
        self.battery -= self.cfg.battery_clean
        return got

    def _move_toward(self, goals):
        p = self.path(self.pos, goals)
        if not p or len(p) < 2:
            return False
        self.pos = p[1]
        self.battery -= self.cfg.battery_move
        return True

    def step(self, option: str) -> dict:
        """Execute one step of an option. Returns a small event record."""
        ev = {"option": option, "cleaned": 0.0, "moved": False, "stuck": False}
        here_dirty = self.dirt[self.pos] > self.cfg.clean_threshold
        if option in ("clean_quiet", "clean_full"):
            room = self.room()
            if room in self.cfg.assigned and here_dirty:
                ev["cleaned"] = self._clean(0.5 if option == "clean_quiet" else 1.0)
            elif room in self.cfg.assigned and self.cells(room):
                ev["moved"] = self._move_toward(self.cells(room))
            else:
                ev["stuck"] = True
        elif option == "next_room":
            nxt = self.next_assigned_room()
            if nxt is None:
                ev["stuck"] = True
            elif self.room() == nxt and here_dirty:
                ev["cleaned"] = self._clean(1.0)
            else:
                ev["moved"] = self._move_toward(self.cells(nxt))
        elif option == "detour":
            vis = self.visible_unassigned()
            if not vis:
                ev["stuck"] = True
            elif self.pos in vis:
                ev["cleaned"] = self._clean(1.0)
            else:
                best = max(vis, key=lambda rc: self.dirt[rc] - 0.03 * abs(rc[0] - self.pos[0]) - 0.03 * abs(rc[1] - self.pos[1]))
                ev["moved"] = self._move_toward([best])
        elif option == "dock":
            if self.pos == DOCK:
                self.battery = min(1.0, self.battery + self.cfg.charge_rate)
            else:
                ev["moved"] = self._move_toward([DOCK])
        self.battery = max(0.0, self.battery)
        self.t += 1
        ev["cell"] = self.pos
        return ev

    def charge_plan(self):
        """Cost of an explicit (top-level) decision: planning uses computing power and battery."""
        self.planning_calls += 1
        self.battery = max(0.0, self.battery - self.cfg.battery_plan)

    def layout(self) -> dict:
        return {"rows": LAYOUT, "rooms": ROOMS, "doors": DOORS, "dock": DOCK,
                "assigned": list(self.cfg.assigned), "tempting": self.cfg.tempting,
                "quiet_zone": self.cfg.quiet_zone, "dirt": np.round(self.initial_dirt, 3).tolist()}
