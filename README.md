# Instruction Synthesis Engine

Evolutionary generation of high-complexity instruction datasets from instruction-tuned language models.

Generates advanced instruction-response pairs suitable for fine-tuning language models, based on evolutionary mutation strategies. This approach enables creation of complex, diverse datasets without relying on proprietary models or violating terms of service.

- **Input:** Language model and optional seed instructions (or document corpus)
- **Output:** Set of evolved instructions with corresponding model responses

Based on evolutionary instruction synthesis methodology from https://arxiv.org/abs/2304.12244

## Overview

The system uses iterative mutation strategies to transform seed instructions into increasingly complex variants:
- Starting instruction: "What's trending in science & technology?"
- Example evolved output: "As a researcher in the field of artificial intelligence (AI) and healthcare, you have been tasked with conducting a comprehensive study of the potential of AI in healthcare. Your research must be based on sources that are at least 10 years old, and you must use a minimum of 15 academic sources..."

## Setup

### Environment Configuration

Create a Python 3.10+ virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Generate Instruction Dataset

Update model and configuration in `instruction_synthesizer.py`, then execute:

```bash
python instruction_synthesizer.py
```

Output: JSON file named `synthesis_output.{id}.json` containing instruction-response pairs

## Mutation Strategies

The synthesizer applies multiple evolution strategies:

1. **Fresh Start** - Generate random instructions from vocabulary
2. **Complicate** - Increase complexity of existing instructions
3. **Add Constraints** - Introduce additional requirements
4. **Deepen** - Expand scope and depth
5. **Concretize** - Make instructions more specific
6. **Increase Reasoning** - Require multi-step thinking
7. **Switch Topic** - Change domain while maintaining difficulty

## Key Components

- **InstructionSynthesizer**: Main orchestrator for instruction evolution
- **LocalModelClient**: Interface for local HuggingFace model inference
- **RemoteModelClient**: Interface for Gradio-hosted remote models
- **EvolutionStrategy**: Enumeration of mutation types

## Current Limitations

- Requires instruct-tuned model for quality outputs
- Can be slow depending on model size and available hardware
- Response generation quality depends on underlying LLM capability
- Single-document processing (corpus support planned)

## Future Enhancements

- Complexity level controls
- Batch processing optimization
- Support for diverse instruction/input formats
- Integration with open-source model pipelines
- Performance optimization for larger datasets

## License

This implementation extends the research presented in https://arxiv.org/abs/2304.12244
