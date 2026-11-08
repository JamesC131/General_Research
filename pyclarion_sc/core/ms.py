"""
Motivational Subsystem (MS): drives at the bottom level.

Drive strength (Gomez & Sun 2024, Eq. 5; Gomez & Sun 2025, Eq. 2):

    ds_d = stimuluslevel_d * deficit_d

Deficit reduction over time on a drive-satisfying task (2024, Eq. 6):

    delta_d = -k_d * time        deficit_d(t) = deficit_d(0) - k_d * time

stimuluslevel is state-specific (how relevant the current situation is to the
drive); deficit is agent-specific (individual or cultural predisposition).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Drive:
    name: str
    stimulus_level: float = 0.0
    deficit: float = 0.0
    floor: float = 0.0           # deficits never go below this value
    initial_deficit: float | None = None

    def __post_init__(self):
        if self.initial_deficit is None:
            self.initial_deficit = self.deficit

    @property
    def strength(self) -> float:
        return self.stimulus_level * self.deficit


# Sun's primary drives (Sun 2009, 2016). Handy defaults for new models/robots.
LOW_LEVEL_DRIVES = ("food", "water", "sleep", "avoiding_danger", "avoiding_unpleasant_stimuli",
                    "reproduction", "conservation")
HIGH_LEVEL_DRIVES = ("affiliation_belongingness", "recognition_achievement", "dominance_power",
                     "autonomy", "deference", "similance", "fairness", "honor", "nurturance",
                     "curiosity")


@dataclass
class MotivationalSubsystem:
    drives: dict[str, Drive] = field(default_factory=dict)

    def add(self, name: str, stimulus_level: float = 0.0, deficit: float = 0.0, floor: float = 0.0) -> Drive:
        self.drives[name] = Drive(name, stimulus_level, deficit, floor)
        return self.drives[name]

    def __getitem__(self, name: str) -> Drive:
        return self.drives[name]

    def strengths(self) -> dict[str, float]:
        """Eq. 5 for every drive."""
        return {n: d.strength for n, d in self.drives.items()}

    def set_stimulus(self, **levels: float) -> None:
        """Update state-specific stimulus levels from the current perception."""
        for name, value in levels.items():
            self.drives[name].stimulus_level = value

    def reduce_deficit(self, name: str, k: float, time_on_task: float) -> float:
        """
        Eq. 6: deficit at time t = initial deficit - k * time on task.
        Returns the deficit reduction delta_d (a negative number or 0).
        """
        d = self.drives[name]
        new = max(d.initial_deficit - k * time_on_task, d.floor)
        delta = new - d.initial_deficit
        d.deficit = new
        return delta

    def snapshot(self) -> dict:
        return {n: {"stimulus_level": d.stimulus_level, "deficit": d.deficit, "strength": d.strength}
                for n, d in self.drives.items()}
