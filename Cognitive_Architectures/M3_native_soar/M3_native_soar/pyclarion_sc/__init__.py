"""
pyclarion-sc: the Clarion cognitive architecture for modeling self-control.

Subsystems
    MotivationalSubsystem   (MS)   drives: strength = stimulus level x deficit
    MetacognitiveSubsystem  (MCS)  goal setting; utility-based explicitness
    ActionCenteredSubsystem (ACS)  top-level rules + bottom-level networks
    WorkingMemory           (NACS) LSTM working memory

Agent
    ClarionAgent: perceive -> drives -> goal + explicitness -> action.
    Works with any Environment that has reset() and step(action).

Experiments (Gomez & Sun 2024, 2025)
    pyclarion_sc.experiments.schmeichel2007
    pyclarion_sc.experiments.friese2008
    pyclarion_sc.experiments.myrseth2009   (two interpretations)
"""
from .agent import ClarionAgent, Environment, StepRecord
from .core.acs import ActionCenteredSubsystem, BottomLevel, Rule, TopLevel, make_mlp_q
from .core.choice import boltzmann_choice, boltzmann_probs
from .core.mcs import ExplicitnessDecision, Goal, MetacognitiveSubsystem
from .core.ms import Drive, MotivationalSubsystem
from .core.nacs import WorkingMemory
from .core.nets import LSTM, MLP

__version__ = "0.5.0"
__all__ = ["ClarionAgent", "Environment", "StepRecord", "ActionCenteredSubsystem", "BottomLevel", "Rule",
           "TopLevel", "make_mlp_q", "boltzmann_choice", "boltzmann_probs", "ExplicitnessDecision", "Goal",
           "MetacognitiveSubsystem", "Drive", "MotivationalSubsystem", "WorkingMemory", "LSTM", "MLP"]
