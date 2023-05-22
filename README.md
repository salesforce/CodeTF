
    
<p align="center">
    <br>
    <img src="assets/logo.png" width="500"/>
    <br>
<p>
<div align="center">
  <a href="https://opensource.org/licenses/BSD-3-Clause">
  <img alt="license" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"/>
  </a>
   <a href="https://www.python.org/downloads/release/python-380/">
  <img alt="license" src="https://img.shields.io/badge/python-3.8+-blue.svg"/>
  </a> 
  
# CodeTF - A Comprehensive Transformer-based Library for Code LLM & Code Intelligence

<!-- 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/bdqnghi/CodeTF_personal/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->
 </div>   
    
## Table of Contents
  - [Introduction](#introduction)
  - [Installation](#installation)
  - [Getting Started](#getting-started)
  - [Other Utilities](#other-utils)
  - [License](#license)

## Introduction
CodeTF is a one-stop Python library for code intelligence tasks (AI4Code), provides a seamless interface for training and inferencing on code intelligence tasks like code summarization, translation, and generation. It aims to facilitate easy integration of cutting-edge language models into real-world applications.

In addition to the core tasks, CodeTF offers utilities for code manipulation across various languages, including easy extraction of code attributes. Using tree-sitter as its core parser, it enables parsing of attributes such as function names, comments, and variable names. Pre-built libraries for numerous languages are provided, eliminating the need for complicated parser setup. CodeTF thus ensures a user-friendly and accessible environment for code intelligence tasks.

The current version of the library offers:

- **Fast Model Serving**: We support an easy-to-use interface for rapid inferencing with pre-quantized models (int4, int8, int16, float16, mixed int8_float16).
- **Fine-Tuning Your Own Models with Custom Datasets**: We provide an API for quickly fine-tuning your own LLMs for code using SOTA techniques for parameter-efficient fine-tuning (HuggingFace PEFT).
- **Supported Tasks**: nl2code, code summarization, code completion, code translation, code refinement, clone detection, defect prediction.
- **Datasets+**: We have preprocessed well-known benchmarks (Human-Eval, MBPP, CodeXGLUE, APPS) and offer an easy-to-load feature for these datasets.
- **Pretrained Models**: We supply pretrained checkpoints of state-of-the-art foundational language models of code (CodeT5, CodeGen, CodeT5+).
- **Fine-Tuned Models**: We furnish fine-tuned checkpoints for 8+ downstream tasks.
- **Utility to Manipulate Source Code**: We provide utilities to easily manipulate source code, such as user-friendly AST parsers in 15+ programming languages.

The following table shows the supported models with sizes and the tasks that the models support. This is a continuing effort and we are working on further growing the list.
    
| Model      | Size                                      | Tasks                                                                                      |
|------------|-------------------------------------------|--------------------------------------------------------------------------------------------|
| CodeT5     | Small (125M), Medium (220M), Large (770M) | Pretrained, Code Sum, Code Generation, <br> Code Refinement, Defect Prediction, Clone Detection |
| CodeT5+    | 220M, 770M, 2B, 6B, 16B                   | Pretrained                                                                                 |
| CodeGen    | 350M, 2B, 6B, 16B                         | Pretrained                                                                                 |
| SantaCoder | 1.1B                                      | Pretrained                                                                                 |
| StarCoder  | 15.5B                                     | Pretrained                                                                                 |
| GPT        | j (1.3B), j (6B), Neox (20B)             | Pretrained                                                                                 |
| GPT-Neo    | 1.3B                                      | Pretrained                                                                                 |
| BLOOM      | 560M, 1.1B, 1.7B, 3B, 7.1B                | Pretrained                                                                                 |


## Quick Start
### Install CodeTF:

1. (Optional) Creating conda environment

```bash
conda create -n codetf python=3.8
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

### Inferencing Pipeline
    
The function ``load_model_pipeline()`` is an important function that loads our supported models and tasks. Below is an example on how to use this function to load ``codet5`` models and perform inference on specific tasks (code translation and code summarization in this case). There are a few notable arguments that need to consider:
-  ``model_name``: the name of the model, currently support ``codet5`` and ``causal-lm``. 
-  ``model_type``: type of model for each model name, e.g. ``base``, ``codegen-350M-mono``, ``j-6B``, etc.
-  ``load_in_8bit``: inherit the ``load_in_8bit" feature from [Huggingface Quantization](https://huggingface.co/docs/transformers/main/main_classes/quantization).
-  ``weight_sharding``: our advance feature that leverate [HuggingFace Sharded Checkpoint](https://huggingface.co/docs/accelerate/v0.19.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch) to split a large model in several smaller shards in different GPUs.
    
```python
from codetf.models import load_model_pipeline

translation_model = load_model_pipeline(model_name="codet5", task="translate-cs-java",
            model_type="base", is_eval=True,
            load_in_8bit=True, weight_sharding=False)

summarization_model = load_model_pipeline(model_name="codet5", task="sum-python",
            model_type="base", is_eval=True,
            load_in_8bit=True, weight_sharding=False)

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

translated_code_snippets = translation_model.predict([code_snippets])

print(translated_code_snippets)

summaries = summarization_model.predict([code_snippets])
print(summaries)
```

## Loading Preprocessed Data

We provide ``Dataset`` class for well-known datasets, including CodeXGLUE, Human Eval, MBPP, APPS. Below is an example of how to load the CodeXGLUE dataset.  

```python
from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=True)
dataset = CodeXGLUEDataset(tokenizer=tokenizer)
train, test, validation = dataset.load(subset="text-to-code")
```
    
    
## Training Custom Model Using Our Dataloader and Trainer
We also provide the users the ability to fine-tune their own LLMs for code using our utility.  Below is an example that use the CausalLMTrainer to fine-tune a code summarization model based on the CodeXGLUE dataset. First, the ``model_class`` is the class that contain the supported models in our pipeline. Next, the ``dataloader`` is an instance from our ``CodeXGLUEDataset``. Then we can load the dataset part that has been processed into appropriate format for training. Finally, the datasets are fed into the ``CausalLMTrainer`` with other parameters to fine-tune a custom model.
        
    
```python
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
import torch
from codetf.trainer.causal_lm_trainer import CausalLMTrainer
from codetf.data_utility.codexglue_dataloader import CodeXGLUEDataLoader
from codetf.models import load_model_pipeline
from codetf.performance.evaluate import Evaluator

model_class = load_model_pipeline(model_name="causal-lm", task="pretrained",
                model_type="codegen-350M-mono",
                quantize=None, quantize_algo="bitsandbyte")


dataloader = CodeXGLUEDataLoader(tokenizer=model_class.get_tokenizer())
train_dataset, test_dataset, val_dataset = dataloader.load_codexglue_code_to_text_dataset()

evaluator = Evaluator(metric="bleu", tokenizer=model_class.tokenizer)

# peft can be in ["lora", "prefixtuning"]
trainer = CausalLMTrainer(train_dataset=train_dataset, 
                        validation_dataset=val_dataset, 
                        peft=None,
                        pretrained_model_or_path=model_class.get_model(),
                        tokenizer=model_class.get_tokenizer())
trainer.train()
# trainer.evaluate(test_dataset=test_dataset)
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


## Technical Report and Citing CodeTF
You can find more details in our [technical report](https://arxiv.org/abs/2209.09019).

If you're using CodeTF in your research or applications, please cite using this BibTeX:
```bibtex
@misc{nghi2023codetf,
      title={CodeTF: A Transformer-based Library for CodeLLM & Code Intelligence}, 
      author={Nghi D. Q. Bui, Henry Le, Yue Wang, Akhilesh Deepak Gotmare, Junna Li, Steven Hoi.},
      year={2023},
      eprint={2209.09019},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact us
If you have any questions, comments or suggestions, please do not hesitate to contact us at codetf@salesforce.com.

## License
[BSD 3-Clause License](LICENSE.txt)
