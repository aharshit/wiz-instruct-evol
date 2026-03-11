"""Utility functions for instruction synthesis"""

import pandas as pd
from datasets import Dataset, DatasetDict
import markdown
from bs4 import BeautifulSoup


def convert_markdown_to_plaintext(md_content, convert_enabled=True):
    """
    Helper function to convert markdown formatted content to plain text.
    
    Args:
        md_content: Markdown formatted string
        convert_enabled: Whether to perform conversion
        
    Returns:
        Plain text string
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
    
    Args:
        instruction_list: List of instruction strings
        
    Returns:
        DatasetDict with 'train' split
    """
    dataframe = pd.DataFrame({'text': instruction_list})
    dataset_dict = DatasetDict()
    dataset_dict['train'] = Dataset.from_pandas(dataframe)
    return dataset_dict
