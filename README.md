# Contextual Linear Activation Steering of Language Models

This repo is the official implementation of Contextual Linear Activation Steering (CLAS), an activation steering method that performs context-dependent steering.

<details>
<summary>What is CLAS?</summary>

> Activation steering is a method that adds interpretable directions (extracted from activation probes) to the internal activations of a model (e.g., an LLM) at inference, enabling interpretable control over its behavior. However, most approaches scale these directions using fixed steering coefficients tuned via grid search, which can be slow to tune and result in suboptimal performance as the same scale is applied to all inputs. CLAS addresses both of these issues by replacing fixed coefficients with learned vectors that map model activations to context-dependent steering coefficients, improving overall steering accuracy and reliability. It is a general-purpose method that can steer many behaviors, such as steering the model toward step-by-step reasoning, making the model take on a specific persona, and reducing toxicity in model outputs.

</details>

Compared to prior activation steering methods, CLAS offers several practical benefits:

- No grid search to tune coefficients
- Context-dependent steering coefficients
- Works with any existing steering vectors (e.g., probes)
- Supports steering multiple directions simultaneously
- Faster tuning and higher accuracy while remaining interpretable

---

## Installation

Requirements:
- Python >= 3.10

```bash
pip install -r requirements.txt
```

---

## Usage

Example commands to reproduce our experiments. Before running the commands below, set `ROOT` in `scripts/__import__.py` to the path of this project. Then run the following commands from the `scripts` directory.

### Probing
    sh slum/Llama-3.2-1B-Instruct/01_probe.sh
Train activation probes to extract steering directions (results written to `directions` by default). The default probe is [RFM](https://github.com/aradha/recursive_feature_machines).

### Tuning
    SLURM_ARRAY_TASK_ID=2 sh slum/Llama-3.2-1B-Instruct/03_tunep.sh
Learn a mapping from model activations to context-dependent steering coefficients (results written to `coefficients` by default).

### Generation
    SLURM_ARRAY_TASK_ID=3 sh slum/Llama-3.2-1B-Instruct/04_generatep.sh
Generate steered model completions (results written to `coefficients` by default).

### Evaluation
    python 05_judge.py
Evaluate the steered outputs (e.g., reasoning, persona, toxicity, etc.). Note: OpenAI API key is required to run this script.