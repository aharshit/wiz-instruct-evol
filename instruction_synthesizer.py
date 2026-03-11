"""
Instruction Synthesis Engine - Main Entry Point
Automated generation of complex instructions from seed prompts and LLM models
Based on evolutionary mutation strategies for instruction complexity
"""

from synthesizer import InstructionSynthesizer, LocalModelClient, RemoteModelClient


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
