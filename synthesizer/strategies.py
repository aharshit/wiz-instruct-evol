"""Evolution strategies for instruction mutation"""

from enum import Enum


class EvolutionStrategy(Enum):
    """Mutation strategies for instruction evolution"""
    FRESH_START = 0
    ADD_CONSTRAINTS = 1
    DEEPEN = 2
    CONCRETIZE = 3
    INCREASE_REASONING = 4
    COMPLICATE = 5
    SWITCH_TOPIC = 6
