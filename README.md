# ASR Error Correction Using Large Language Models

## Overview

This project investigates the use of **Large Language Models (LLMs)** for enhancing **Automatic Speech Recognition (ASR)** systems by reducing **Word Error Rate (WER)**. By leveraging techniques like **Low-Rank Adaptation (LoRA)** and **Evolutionary Prompt Optimization (EvoPrompt)**, we demonstrate how compact and large LLMs can be optimized for error correction across noisy, accented, and domain-specific environments.

---

## Key Features

- **ASR Error Correction**: Post-processing top-N hypotheses from ASR systems.
- **Low-Rank Adaptation (LoRA)**: Efficient fine-tuning with low computational overhead.
- **EvoPrompt Optimization**: Automated prompt tuning using evolutionary algorithms.
- **Additional Methods**: TokenLimit, Repetition Penalty, and Word Filtering for refinement.
- **Model Evaluation**: Comparing TinyLlama, Qwen, GPT-3.5, and LLaMA models.

---

## Experimental Setup

### Dataset

- **Wall Street Journal (WSJ)** dataset from HP-v0:
  - **Train**: 37,514 utterances (WSJ train-si284 split).
  - **Test**: 503 utterances (dev93) + 333 utterances (eval92).

### Models

- **TinyLlama-1.1B-Chat-v1.0**: Compact, resource-efficient LLM.
- **Qwen-2.5B-Instruct**: Mid-sized, instruction-following model.
- **LLaMA-3.2-3B-Instruct**: Scalable, general-purpose model.
- **OpenAI GPT-3.5 Turbo**: Large-scale conversational model.

### Techniques

- **Plain model performance**: Baseline results without optimization.
- **LoRA Fine-Tuning**: Injecting trainable low-rank matrices for efficient model updates.
- **EvoPrompt**: Iterative prompt refinement using Genetic Algorithms (GA).
- **Full Tuning**: Combining LoRA, EvoPrompt, and additional post-processing methods.

---

## Results

| Model           | Optimization       | WER (%) |
|------------------|--------------------|---------|
| **ASR Baseline** | None              | **5.31** |
| TinyLlama        | None              | 285.89  |
| TinyLlama        | LoRA              | 87.57   |
| TinyLlama        | EvoPrompt         | 273.51  |
| TinyLlama        | LoRA+EvoPrompt    | 82.65   |
| TinyLlama        | Full Tuning       | **7.47** |
| GPT-3.5          | None              | 44.90   |
| GPT-3.5          | EvoPrompt         | 33.96   |
| Qwen-2.5-7B      | None              | 1503.20 |
| Qwen-2.5-7B      | LoRA              | 1371.20 |
| Qwen-2.5-7B      | EvoPrompt         | 1411.30 |
| Qwen-2.5-7B      | Full Tuning       | **6.25** |
| Llama3.2-3B      | None              | 1213.00 |
| Llama3.2-3B      | Full Tuning       | **6.90** |


---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/KOURENYILI/Error-Correction-By-using-LLM.git
   cd Error-Correction-By-using-LLM
   ```

2. Install dependencies in each folder:
   ```bash
   pip install -r requirements.txt
   ```
   ---

## Architecture

The workflow integrates ASR systems, N-best hypotheses, and LLMs with the following pipeline:

1. **ASR System**: Generates top-N transcriptions.
2. **LoRA Fine-Tuning**: Adapts LLM to ASR-specific tasks.
3. **EvoPrompt Optimization**: Refines model prompts.
4. **Post-Processing**: Applies TokenLimit, Repetition Penalty, and Word Filtering.

---

## Future Directions

- **Performance Gap Analysis**: Address discrepancies with state-of-the-art systems.
- **Adaptive Optimization**: Dynamically adjust tuning for varied acoustic environments.
- **Resource Efficiency**: Explore better trade-offs for smaller models.

---

