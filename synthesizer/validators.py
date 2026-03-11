"""Validation and cleaning logic for generated instructions"""

from typing import Tuple


def clean_instruction_output(raw_output: str) -> str:
    """
    Clean and normalize instruction output from LM.
    
    Args:
        raw_output: Raw output from language model
        
    Returns:
        Cleaned instruction string
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


def validate_instruction_evolution(
    original: str, 
    evolved: str,
    base_template: str = ""
) -> Tuple[bool, str]:
    """
    Validate that instruction evolution was successful.
    
    Args:
        original: Original instruction
        evolved: Evolved instruction
        base_template: Base template string for leak detection
        
    Returns:
        Tuple of (is_valid, reason)
    """
    if original == evolved:
        return False, "unchanged"
    
    if evolved.count('\n') > evolved.count(" ") * 2:
        return False, "excessive_newlines"
    
    if evolved.count('\n') == evolved.count("- ") > 10:
        return False, "excessive_list_items"
    
    if base_template and base_template in evolved:
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
