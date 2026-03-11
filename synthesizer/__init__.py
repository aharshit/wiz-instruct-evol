"""Instruction Synthesis Engine Package"""

from .core import InstructionSynthesizer
from .models import LocalModelClient, RemoteModelClient
from .strategies import EvolutionStrategy
from .utils import convert_markdown_to_plaintext, build_dataset_from_list
from .validators import clean_instruction_output, validate_instruction_evolution

__all__ = [
    'InstructionSynthesizer',
    'LocalModelClient',
    'RemoteModelClient',
    'EvolutionStrategy',
    'convert_markdown_to_plaintext',
    'build_dataset_from_list',
    'clean_instruction_output',
    'validate_instruction_evolution',
]
