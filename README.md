
    
<p align="center">
    <br>
    <img src="assets/logo.png" width="200"/>
    <br>
<p>

    
# CodeTF - A Comprehensive Library for Revolutionizing Code Intelligence
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/bdqnghi/CodeTF_personal/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
    
## Table of Contents
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
  - [Other Utilities](#other-utils)
  - [License](#license)

## Introduction
CodeTF is a state-of-the-art deep learning library in Python designed to provide a comprehensive interface for training and inferencing on code intelligence tasks, collectively known as AI4Code. This library encompasses a broad spectrum of tasks, including code summarization, code translation, code generation, and more. Our primary goal is to equip researchers and engineers with a one-stop solution that enables them to seamlessly explore the potential of cutting-edge language models for code. With its intuitive and user-friendly interface, we aspire to facilitate the integration of AI4Code into real-world applications with minimal effort.

As an all-inclusive code intelligence toolkit, CodeTF also offers utilities that allow for effortless manipulation of source code across various programming languages. These utilities enable users to extract code attributes such as function names, comments, identifiers, and variable names. To accomplish this, an Abstract Syntax Tree (AST) parser is essential, and our library leverages tree-sitter as its core parser. To enhance accessibility for our users, we have pre-built tree-sitter libraries into .so files for immediate use, covering programming languages like Bash, C#, C++, C, CSS, ELM, Go, Haskell, HTML, Java, JavaScript, Kotlin, Lua, PHP, Python, Ruby, Rust, Scala, Solidity, and SFApex. This eliminates the need for users to set up these parsers, which can often be challenging. With CodeTF, users can instantly utilize our utilities with ease.

The current version of the library offers:

- **Fast Model Serving**: We support an easy-to-use interface for rapid inferencing with pre-quantized models (int4, int8, int16, float16, mixed int8_float16).
- **Fine-Tuning Your Own Models with Custom Datasets**: We provide an API for quickly fine-tuning your own LLMs for code using SOTA techniques for parameter-efficient fine-tuning (HuggingFace PEFT).
- **Supported Tasks**: nl2code, code summarization, code completion, code translation, code refinement, clone detection, defect prediction.
- **Datasets+**: We have preprocessed well-known benchmarks (Human-Eval, MBPP, CodeXGLUE, APPS) and offer an easy-to-load feature for these datasets.
- **Pretrained Models**: We supply pretrained checkpoints of state-of-the-art foundational language models of code (CodeT5, CodeGen, CodeT5+).
- **Fine-Tuned Models**: We furnish fine-tuned checkpoints for 8+ downstream tasks.
- **Utility to Manipulate Source Code**: We provide utilities to easily manipulate source code, such as user-friendly AST parsers in multiple languages.

    

## Available Models & Tasks
The following table shows the available models with their checkpoints and the supported quantized versions. This is a continuing effort and we are working on further growing the lis.
    

| Models  | Checkpoints       | int4 | int8 | int16 | float16 |
|---------|-------------------|------|------|-------|---------|
| CodeT5  | Pretrained        | ✓    | ✓    | ✓     | ✓       |
|         | Code Generation   |      | ✓    | ✓     | ✓       |
|         | Code Summarization|      | ✓    | ✓     | ✓       |
|         | Code Completion   |      | ✓    | ✓     | ✓       |
|         | Code Refinement   |      | ✓    | ✓     | ✓       |
| CodeT5+ | Stage-1           |      |      |       |         |
|         | Stage-2           |      |      |       |         |
| CodeGen | Pretrained        |      | ✓    | ✓     | ✓       |



    
## Feature Comparison
    
The following table shows the features comparison between CodeTF and other libraries, such as NaturalCC and HuggingFace Transformers. It is important to note that HuggingFace Transformers (HF-T) is a comprehensive library encompassing state-of-the-art language models and utilities for multiple research domains. The comparison provided in this Table focuses solely on the features related to the code domain, highlighting areas where HuggingFace Transformers may lack certain functionality. This is also a continuing effort and we are working on further growing the list.

|                                            | CodeTF (Ours) | NaturalCC | HuggingFace-Transformers |
|--------------------------------------------|---------------|-----------|------|
| Unified Model and Dataset Interface        | ✓             |           |      |
| Unified Interface for Parameter-Efficient Fine-Tuning | ✓ |           |      |
| Unified Code Utility Interface for Multiple Programming Languages | ✓ |  |      |
| Unified Interface for Evaluation           | ✓             |           |      |
| Modular Library Design                     | ✓             |           | ✓    |
| Pretrained Model Checkpoints               | ✓             | ✓         | ✓    |
| Task-specific Finetuned Model Checkpoints  | ✓             | ✓         | ✓    |
| **Tasks**                                  |               |           |      |
| Code Summarization                         | ✓             | ✓         | ✓    |
| Code Generation                            | ✓             | ✓         | ✓    |
| Code Completion                            | ✓             | ✓         | ✓    |
| Code Refinement                            | ✓             | ✓         | ✓    |
| Defect Prediction                          | ✓             |           |      |
| **Datasets**                               |               |           |      |
| Human Eval                                 | ✓             |           |      |
| MBPP                                       | ✓             |           |      |
| APPS                                       | ✓             |           |      |
| Py150                                      | ✓             | ✓         |      |
| CodeXGLUE                                  | ✓             | ✓         |      |


## Getting Started

    
    
### Install CodeTF:

1. (Optional) Creating conda environment

```bash
conda create -n lavis python=3.8
conda activate codetf
```

2. install from [PyPI](https://pypi.org/project/salesforce-codetf/)
```bash
pip install codetf
```
    
3. Or, for development, you may build from source

```bash
git clone https://github.com/salesforce/CodeTF.git
cd CodeTF
pip install -e .
```

### Example Usage to make inference
```python
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = load_model(name="codet5_summarization", model_type="base", is_eval=True, device=device)
code_snippets = """
    void bubbleSort(int arr[])
    {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++)
            for (int j = 0; j < n - i - 1; j++)
                if (arr[j] > arr[j + 1]) {
                    // swap arr[j+1] and arr[j]
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
    }

"""
summarizations = model.predict([code_snippets])
print(summarizations)
```

### Preprocessed Datasets
```python
from codetf.datasets import load_dataset
codexglue_codesum = load_dataset("codexglue_codesum")
print(codexglue_codesum.keys())
# dict_keys(['train', 'val', 'test'])
print(len(codexglue_codesum["train"]))
print(codexglue_codesum["train"][0])
```

### Fine-tuning Your Own Model on a Generative Task
```python
from codetf.datasets import load_dataset
from codetf.trainer.codet5_seq2seq_trainer import CodeT5Seq2SeqTrainer

codexglue_codesum = load_dataset("codexglue_codesum")

codet5_trainer = CodeT5Seq2SeqTrainer()

```

## Code utilities
### AST Parser in Multiple Languages

Below is an example to parse Apex code into an AST.
```python
from codetf.code_ultilities import load_parser

sfapex_parser = load_parser(language="apex")

code_snippets = """
    void bubbleSort(int arr[])
    {
        int n = arr.length;
        for (int i = 0; i < n - 1; i++)
            for (int j = 0; j < n - i - 1; j++)
                if (arr[j] > arr[j + 1]) {
                    // swap arr[j+1] and arr[j]
                    int temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
    }

"""
ast = sfapex_parser.parse(code_snippets)
```

Then you can traverse the tree using the interface from ```py-tree-sitter```
```
root_node = ast.root_node
assert root_node.type == 'module'
assert root_node.start_point == (1, 0)
assert root_node.end_point == (3, 13)
```

### Extract Code Attributes

We also provide interface to extract the code attributes easily. Below is the sample to extract the function name of a python function:

```python
from codetf.code_ultilities import load_code_attributes_extractor

extractor = load_code_attributes_extractor(language="python")

code_snippets = """
    def add_two_numbers(a, b):
    {
        return a + b
    }

"""

function_name = extractor.extract_function_name(code_snippets)
print(function_name)
```

This will print ```add_two_numbers``` as the function name.
