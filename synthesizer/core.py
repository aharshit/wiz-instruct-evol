"""Core instruction synthesizer engine"""

import time
import json
import uuid
import pickle
import os
from typing import List

import numpy as np

from .strategies import EvolutionStrategy
from .utils import build_dataset_from_list
from .validators import clean_instruction_output, validate_instruction_evolution


class InstructionSynthesizer:
    """
    Generate complex instructions through iterative mutation strategies.
    Based on: https://arxiv.org/abs/2304.12244
    """
    
    def __init__(
        self,
        language_model=None,
        initial_prompts: List[str] = None,
        target_count: int = 10,
        min_length_bytes: int = 512,
        max_length_bytes: int = 1024,
        verbose: bool = False,
    ):
        """
        Initialize instruction synthesizer.
        
        Args:
            language_model: Model pipeline for generation
            initial_prompts: Seed prompts for evolution
            target_count: Number of final instructions to generate
            min_length_bytes: Minimum instruction length
            max_length_bytes: Maximum instruction length
            verbose: Enable verbose output
        """
        self.language_model = language_model
        self.target_count = target_count
        self.verbose = verbose
        self.instruction_bank = []
        self.initial_prompts = initial_prompts
        self.evolution_queue = []
        self.synthesized_instructions = []
        self.synthesized_responses = []
        self.min_length_bytes = min_length_bytes
        self.max_length_bytes = max_length_bytes
        
        # Load vocabulary for random instruction generation
        self._load_vocabulary()
        self._initialize_strategy_templates()

    def _load_vocabulary(self):
        """Load English nouns for random instruction generation"""
        with open("english-nouns.txt") as f:
            self.vocabulary_terms = f.readlines()

    def _initialize_strategy_templates(self):
        """Setup mutation strategy prompt templates"""
        self.strategy_templates = dict()
        self.strategy_templates['base'] = ""
        
        np.random.seed(None)
        
        self.strategy_templates[EvolutionStrategy.FRESH_START] = \
            self.strategy_templates['base'] + \
"""Write one question or request containing one or more of the following words: <PROMPT>"""

        self.strategy_templates[EvolutionStrategy.COMPLICATE] = \
            self.strategy_templates['base'] + \
"""Rewrite #Given Prompt# to make it slightly more complicated, and create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.strategy_templates[EvolutionStrategy.ADD_CONSTRAINTS] = \
            self.strategy_templates['base'] + \
"""Add a few more constraints or requirements to #Given Prompt#, and create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.strategy_templates[EvolutionStrategy.DEEPEN] = \
            self.strategy_templates['base'] + \
"""Slightly increase the depth and breadth of #Given Prompt#, and create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.strategy_templates[EvolutionStrategy.CONCRETIZE] = \
            self.strategy_templates['base'] + \
"""Make #Given Prompt# slightly more concrete, and create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.strategy_templates[EvolutionStrategy.INCREASE_REASONING] = \
            self.strategy_templates['base'] + \
"""If #Given Prompt# can be solved with just a few simple thinking processes, rewrite it to explicitly request multi-step reasoning, and create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

        self.strategy_templates[EvolutionStrategy.SWITCH_TOPIC] = \
            self.strategy_templates['base'] + \
"""Rewrite #Given Prompt# by switching the topic, keeping the domain and difficulty level similar, and create #New Prompt#.

#Given Prompt#:
<PROMPT>
"""

    def execute(self):
        """Main execution pipeline"""
        self._prepare_seed_instructions()
        self._evolve_instructions()
        self._save_evolution_artifacts()
        self._generate_responses()
        self._save_final_dataset()

    def _prepare_seed_instructions(self):
        """Initialize seed instructions from provided data or generate randomly."""
        if isinstance(self.initial_prompts, str) and os.path.exists(self.initial_prompts):
            import pandas as pd
            data_frame = pd.DataFrame(self.initial_prompts)
            if data_frame.shape[1] > 1:
                print("Warning: using only first column")
            self.instruction_bank = data_frame.iloc[:, 0].values.tolist()
            assert self.instruction_bank, "Failed to load instruction data"
        else:
            assert isinstance(self.instruction_bank, list)
            if self.initial_prompts:
                self.instruction_bank = self.initial_prompts
            else:
                # Generate random seed instructions
                for _ in range(self.target_count * 10):
                    num_terms = np.random.choice([1, 2, 3, 4])
                    seed_instruction = self.strategy_templates[EvolutionStrategy.FRESH_START].replace(
                        "<PROMPT>",
                        ", ".join([
                            np.random.choice(self.vocabulary_terms).strip() 
                            for _ in range(num_terms)
                        ])
                    )
                    self.instruction_bank.append(seed_instruction)

    def _evolve_instructions(self):
        """Iteratively evolve instructions through mutations"""
        print(f"Evolving {self.target_count} instructions...")
        assert self.instruction_bank, "Seed instructions required"
        
        begin_time = time.time()
        self.evolution_queue.clear()
        
        for _ in range(self.target_count):
            selected = np.random.choice(self.instruction_bank)
            self.evolution_queue.append(selected)
        
        iteration_idx = 0
        while self._perform_evolution_cycle(iteration_idx):
            print(f"Evolution iteration: {iteration_idx}")
            iteration_idx += 1
        
        end_time = time.time()
        print(f"Generated {len(self.synthesized_instructions)} instructions in {end_time - begin_time:.4f}s")

    def _perform_evolution_cycle(self, cycle_number) -> bool:
        """
        Perform one evolutionary cycle on all instructions.
        
        Args:
            cycle_number: Current iteration number
            
        Returns:
            True if more evolution needed, False if complete
        """
        assert len(self.evolution_queue) == self.target_count
        
        batch_instructions = []
        applied_strategies = []
        
        for instr_idx in range(self.target_count):
            if cycle_number == 0 or "Write one question or request containing" in self.evolution_queue[instr_idx]:
                strategy = EvolutionStrategy.FRESH_START
                applied_strategies.append(strategy)
            else:
                strategy = np.random.choice(EvolutionStrategy)
                applied_strategies.append(strategy)
                
                if strategy == EvolutionStrategy.FRESH_START:
                    self.evolution_queue[instr_idx] = np.random.choice(self.instruction_bank)
            
            original = self.evolution_queue[instr_idx]
            evolved = (
                self.strategy_templates[strategy].replace("<PROMPT>", original) 
                if cycle_number else original
            )
            batch_instructions.append(evolved)

        dataset = build_dataset_from_list(batch_instructions)
        assert dataset['train'].num_rows == len(batch_instructions) == self.target_count
        
        t0 = time.time()
        evolved_results = self.language_model(dataset['train'])
        assert len(evolved_results) == self.target_count
        t1 = time.time()
        print(f"LM inference: {t1 - t0:.4f}s")

        # Post-process and validate evolved instructions
        for batch_idx in range(len(evolved_results)):
            evolved_results[batch_idx] = clean_instruction_output(evolved_results[batch_idx])
            original_instruction = self.evolution_queue[batch_idx]
            
            is_valid, validation_msg = validate_instruction_evolution(
                original_instruction,
                evolved_results[batch_idx],
                self.strategy_templates['base']
            )
            
            if self.verbose:
                print("=" * 50)
                print(f"Original: {original_instruction}")
                print(f"Strategy: {applied_strategies[batch_idx].name}")
                print(f"Evolved: {evolved_results[batch_idx]}")
                print("=" * 50)

            if is_valid:
                if self.max_length_bytes >= len(evolved_results[batch_idx]) >= self.min_length_bytes:
                    self.synthesized_instructions.append(evolved_results[batch_idx])
                    print(f"✓ Accepted - {len(self.synthesized_instructions)} total")
                    self.evolution_queue[batch_idx] = np.random.choice(self.instruction_bank)
                else:
                    self.evolution_queue[batch_idx] = evolved_results[batch_idx]
                    print("⊙ Modified instruction accepted")
            else:
                print(f"✗ Rejected: {validation_msg}")
            print("", flush=True)
        
        return len(self.synthesized_instructions) < self.target_count

    def _save_evolution_artifacts(self):
        """Save intermediate evolution artifacts"""
        with open("synthesized_instructions.pickle", "wb") as f:
            f.write(pickle.dumps(self.synthesized_instructions))

    def _generate_responses(self):
        """Generate responses for all synthesized instructions"""
        print(f"Generating answers for {len(self.synthesized_instructions)} instructions...")
        begin_time = time.time()
        dataset = build_dataset_from_list(self.synthesized_instructions)
        self.synthesized_responses = self.language_model(dataset['train'])
        end_time = time.time()
        print(f"Completed {len(self.synthesized_responses)} responses in {end_time - begin_time:.4f}s")

    def _save_final_dataset(self):
        """Save final instruction-response dataset"""
        with open("generated_responses.pickle", "wb") as f:
            f.write(pickle.dumps(self.synthesized_responses))
        
        dataset_records = []
        for instr_idx in range(len(self.synthesized_instructions)):
            dataset_records.append({
                'input': self.synthesized_instructions[instr_idx],
                'output': self.synthesized_responses[instr_idx],
            })
        
        dataset_id = str(uuid.uuid4())[:4]
        output_filename = f"synthesis_output.{dataset_id}.json"
        
        with open(output_filename, "wt") as f:
            f.write(json.dumps(dataset_records, indent=2))
        
        print(f"Dataset saved to {output_filename}")
