"""
Instruction Data Synthesis Engine
Automated generation of complex instructions from seed prompts and LLM models
Based on evolutionary mutation strategies for instruction complexity
"""

import ast
import time
import json
import uuid
import pickle
import os
from typing import List, Tuple

import pandas as pd
import numpy as np
from enum import Enum

import torch
from datasets import Dataset, DatasetDict
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

import markdown
from bs4 import BeautifulSoup


class EvolutionStrategy(Enum):
    """Mutation strategies for instruction evolution"""
    FRESH_START = 0
    ADD_CONSTRAINTS = 1
    DEEPEN = 2
    CONCRETIZE = 3
    INCREASE_REASONING = 4
    COMPLICATE = 5
    SWITCH_TOPIC = 6


def convert_markdown_to_plaintext(md_content, convert_enabled=True):
    """
    Helper function to convert markdown formatted content to plain text.
    
    :param md_content: Markdown formatted string
    :param convert_enabled: Whether to perform conversion
    :return: Plain text string
    """
    if not convert_enabled:
        return md_content
    assert md_content is not None, "Markdown content cannot be None"
    html = markdown.markdown(md_content)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def build_dataset_from_list(instruction_list):
    """
    Convert list of instructions to a HuggingFace Dataset.
    
    :param instruction_list: List of instruction strings
    :return: DatasetDict with 'train' split
    """
    dataframe = pd.DataFrame({'text': instruction_list})
    dataset_dict = DatasetDict()
    dataset_dict['train'] = Dataset.from_pandas(dataframe)
    return dataset_dict


class RemoteModelClient:
    """Client for accessing remote LLM via Gradio"""
    
    def __init__(self, host_endpoint, **config_params):
        from gradio_client import Client
        self.client = Client(host_endpoint)
        self.config = config_params

    def __call__(self, dataset):
        """Generate responses for dataset using remote endpoint"""
        responses = []
        for record in dataset:
            self.config['instruction_nochat'] = record['text']
            response_data = self.client.predict(
                str(dict(self.config)),
                api_name='/submit_nochat_api'
            )
            responses.append(convert_markdown_to_plaintext(
                ast.literal_eval(response_data)['response']
            ))
        return responses


class LocalModelClient:
    """Local LLM inference handler using HuggingFace transformers"""
    
    def __init__(self, model_name, max_generation_length=None, batch_size=None, **kwargs):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("Loading tokenizer from checkpoint...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        
        print("Loading model weights...")
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        eos_id = model_obj.config.eos_token_id
        del model_obj
        
        print("Initializing generation pipeline...")
        self.inference_pipeline = pipeline(
            model=model_name,
            tokenizer=tokenizer,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            **kwargs,
        )
        print("Pipeline initialization complete.")
        
        self.inference_pipeline.tokenizer.pad_token_id = eos_id
        self.max_generation_length = max_generation_length
        self.batch_size = batch_size

    def __call__(self, dataset):
        """
        Generate responses from dataset
        
        :param dataset: HuggingFace dataset with 'text' column
        :return: List of generated responses
        """
        responses = []
        for idx, output in enumerate(tqdm(
            self.inference_pipeline(
                KeyDataset(dataset, "text"),
                max_new_tokens=self.max_generation_length,
                batch_size=self.batch_size,
            )
        )):
            generated = output[0]["generated_text"]
            generated = generated.replace(dataset[idx]['text'], '').strip()
            responses.append(generated)
        return responses


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
        
        :param language_model: Model pipeline for generation
        :param initial_prompts: Seed prompts for evolution
        :param target_count: Number of final instructions to generate
        :param min_length_bytes: Minimum instruction length
        :param max_length_bytes: Maximum instruction length
        :param verbose: Enable verbose output
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
        """
        Initialize seed instructions from provided data or generate randomly.
        """
        if isinstance(self.initial_prompts, str) and os.path.exists(self.initial_prompts):
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
        Perform one evolutionary cycle on all instructions
        
        :param cycle_number: Current iteration number
        :return: True if more evolution needed, False if complete
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
            evolved_results[batch_idx] = self._clean_instruction_output(evolved_results[batch_idx])
            original_instruction = self.evolution_queue[batch_idx]
            
            is_valid, validation_msg = self._validate_instruction_evolution(
                original_instruction,
                evolved_results[batch_idx]
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

    def _clean_instruction_output(self, raw_output: str) -> str:
        """
        Clean and normalize instruction output from LM
        
        :param raw_output: Raw output from language model
        :return: Cleaned instruction string
        """
        text = raw_output.split("Prompt#:")[-1].strip()
        
        prefixes = ['New Prompt:\n', 'New Prompt: ']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):]
        
        # Remove AI assistant role markers
        ai_markers = [
            ("As an AI assistant, I", "I"),
            ("As an AI language model, I", "I"),
            ("As an AI assistant, you", "You"),
            ("As an AI language model, you", "You"),
            ("As an AI assistant, what", "What"),
            ("As an AI language model, what", "What"),
        ]
        
        for marker, replacement in ai_markers:
            text = text.replace(marker, replacement)
        
        return text.strip()

    def _validate_instruction_evolution(self, original: str, evolved: str) -> Tuple[bool, str]:
        """
        Validate that instruction evolution was successful.
        
        :param original: Original instruction
        :param evolved: Evolved instruction
        :return: (is_valid, reason)
        """
        if original == evolved:
            return False, "unchanged"
        
        if evolved.count('\n') > evolved.count(" ") * 2:
            return False, "excessive_newlines"
        
        if evolved.count('\n') == evolved.count("- ") > 10:
            return False, "excessive_list_items"
        
        if self.strategy_templates['base'] and self.strategy_templates['base'] in evolved:
            return False, "template_leaked"
        
        if "#New Prompt#" in evolved:
            return False, "placeholder_leaked"
        
        if "new prompt" in evolved.lower():
            return False, "prompt_reference_leaked"
        
        if "how can i assist" in evolved.lower():
            return False, "ai_assistant_reference"
        
        if "as an ai" in evolved.lower():
            return False, "ai_role_reference"
        
        if "ai assistant" in evolved.lower():
            return False, "ai_assistant_reference"
        
        if "i'm sorry" in evolved.lower() and "sorry" not in original.lower() and len(evolved) < 400:
            return False, "apologetic_tone"
        
        return True, "valid"

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


def test_load_artifacts():
    """Test function to load and inspect saved artifacts"""
    import pickle
    with open("synthesized_instructions.pickle", "rb") as f:
        loaded_instructions = pickle.loads(f.read())
        print(loaded_instructions)


if __name__ == "__main__":
    model_pipeline = None
    
    try:
        print("Attempting connection to remote model endpoint...")
        model_pipeline = RemoteModelClient(
            "https://3c4be9cfc5b1ebb838.gradio.live",
            instruction='',
            iinput='',
            context='',
            stream_output=False,
            prompt_type='human_bot',
            prompt_dict='',
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=1,
            max_new_tokens=512,
            min_new_tokens=0,
            early_stopping=False,
            max_time=20,
            repetition_penalty=1.07,
            num_return_sequences=1,
            do_sample=False,
            chat=False,
            instruction_nochat='',
            iinput_nochat='',
            langchain_mode='Disabled',
            top_k_docs=4,
            chunk=True,
            chunk_size=512,
            document_choice=['All'],
        )
        print("Remote connection successful")
    except Exception as e:
        print(f"Remote connection failed: {e}")

    if model_pipeline is None:
        print("Loading local model for inference...")
        model_pipeline = LocalModelClient(
            "distilgpt2",
            max_generation_length=128,
            batch_size=1,
            do_sample=True,
        )

    synthesizer = InstructionSynthesizer(
        language_model=model_pipeline,
        initial_prompts=None,
        target_count=8,
        min_length_bytes=256,
        max_length_bytes=1024,
        verbose=True,
    )
    synthesizer.execute()
