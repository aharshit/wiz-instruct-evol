"""Language model clients for instruction generation"""

import ast
import torch
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm

from .utils import convert_markdown_to_plaintext


class RemoteModelClient:
    """Client for accessing remote LLM via Gradio"""
    
    def __init__(self, host_endpoint, **config_params):
        """
        Initialize remote model client.
        
        Args:
            host_endpoint: Gradio endpoint URL
            **config_params: Configuration parameters
        """
        from gradio_client import Client
        self.client = Client(host_endpoint)
        self.config = config_params

    def __call__(self, dataset):
        """
        Generate responses for dataset using remote endpoint.
        
        Args:
            dataset: HuggingFace dataset with instructions
            
        Returns:
            List of generated responses
        """
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
        """
        Initialize local model client.
        
        Args:
            model_name: HuggingFace model identifier
            max_generation_length: Maximum tokens to generate
            batch_size: Batch size for inference
            **kwargs: Additional pipeline arguments
        """
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
        Generate responses from dataset.
        
        Args:
            dataset: HuggingFace dataset with 'text' column
            
        Returns:
            List of generated responses
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
