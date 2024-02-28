
    
<p align="center">
    <br>
    <img src="assets/logo.png" width="500"/>
    <br>
<p>
<div align="center">
  <a href="https://opensource.org/license/apache-2-0/">
  <img alt="license" src="https://img.shields.io/badge/License-Apache%202.0-green.svg"/>
  </a>
   <a href="https://www.python.org/downloads/release/python-380/">
  <img alt="python" src="https://img.shields.io/badge/python-3.8+-yellow.svg"/>
  </a> 
   <a href="https://pypi.org/project/salesforce-codetf/">
  <img alt="downloads" src="https://static.pepy.tech/badge/salesforce-codetf"/>
  </a> 

<a href="https://arxiv.org/pdf/2306.00029.pdf">Technical Report</a>,
<a href="https://opensource.salesforce.com/CodeTF/latest/index.html">Documentation</a>,
<a href="https://github.com/salesforce/CodeTF/tree/main/test_inference">Examples</a>,
    
# CodeTF - A One-stop Transformer Library for State-of-the-art Code LLM

<!-- 
[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/bdqnghi/CodeTF_personal/blob/main/LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) -->
 </div>   
    
## Table of Contents
  - [Introduction](#introduction)
  - [Installation](#installation-guide)
  - [Getting Started](#getting-started)
    - [Inferencing Pipeline](#inferencing-pipeline)
    - [Model Zoo](#model-zoo)
    - [Fine-Tuning Your Own Model](#fine-tuning-pipeline)
    - [Evaluate On Well-Known Benchmarks](#evaluate-on-well-known-benchmarks)
    - [Utilities to Manipulate Source Code Based on AST](#code-utilities)
        - [AST Parser in Multiple Languages](#ast-parser-in-multiple-languages)
        - [Extract Code Attributes](#extract-code-attributes)
        - [Remove Comments](#remove-comments)
  - [Ethical and Responsible Use](#ethical-and-responsible-use) 
  - [License](#license)

## Introduction
CodeTF is a one-stop Python transformer-based library for ***code large language models (Code LLMs)*** and ***code intelligence***, provides a seamless interface for training and inferencing on code intelligence tasks like code summarization, translation, code generation and so on. It aims to facilitate easy integration of SOTA CodeLLMs into real-world applications.

In addition to the core LLMs's features for code, CodeTF offers utilities for code manipulation across various languages, including easy extraction of code attributes. Using tree-sitter as its core AST parser, it enables parsing of attributes such as function names, comments, and variable names. Pre-built libraries for numerous languages are provided, eliminating the need for complicated parser setup. CodeTF thus ensures a user-friendly and accessible environment for code intelligence tasks.

The current version of the library offers:

- **Fast Model Serving**: We support an easy-to-use interface for rapid inferencing with **pre-quantized models** (int8, int16, float16). CodeTF handles all aspects of device management, so users do not have to worry about that aspect. If your model is large, we offer advanced features such as weight sharding across GPUs to serve the models more quickly.
- **Fine-Tuning Your Own Models**: We provide an API for quickly fine-tuning your own LLMs for code using SOTA techniques for **parameter-efficient fine-tuning** (HuggingFace PEFT) on distributed environments.
- **Supported Tasks**: nl2code, code summarization, code completion, code translation, code refinement, clone detection, defect prediction.
- **Datasets+**: We have preprocessed well-known benchmarks (**Human-Eval, MBPP, CodeXGLUE, APPS, etc.**) and offer an easy-to-load feature for these datasets.
- **Model Evaluator**: We provide interface to evaluate models on well-known benchmarks (e.g. Human-Eval) on popular metrics (e.g., pass@k) with little effort (**~15 LOCs**).
- **Pretrained Models**: We supply pretrained checkpoints of state-of-the-art foundational language models of code (CodeBERT, CodeT5, CodeGen, CodeT5+, Incoder, StarCoder, etc.).
- **Fine-Tuned Models**: We furnish fine-tuned checkpoints for 8+ downstream tasks.
- **Utility to Manipulate Source Code**: We provide utilities to easily manipulate source code, such as user-friendly AST parsers (based on tree-sitter) in **15+ programming languages**, to extract important code features, such as function name, identifiers, etc.

The following table shows the supported models with sizes and the tasks that the models support. This is a continuing effort and we are working on further growing the list.
    
| Model        | Size                                                                                                                          | Tasks                                                                                                                                                                                                     |
|--------------|-------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| CodeT5       | Base, Base-multi-sum, Base-translate-cs, Base-translate-java, Base-sum, Base-clone, Base-defect                              | Pretrained, NL to Code, Refine, Translation (CS to Java, Java to CS), Summarization (Python, Go, PHP, JavaScript, Java, Ruby), Clone detection, Defect prediction |
| CodeT5+      | Plus-instruct-16B, Plus-16B, Plus-6B, Plus-2B, Plus-770M-python, Plus-770M, Plus-220M                                      | Pretrained, NL to Code, Refine , Defect prediction |
| CodeGen      | Mono: 350M, 2B, 6B, 1B, 3.7B, 7B, 16B<br>Multi: 350M, 2B, 6B<br>NL: 350M, 2B                                           | Pretrained |
| StarCoder    | 15.5B                                                                                                                         | Pretrained |
| SantaCoder   | 1.1B                                                                                                                          | Pretrained |
| GPT-NeoX     | 20B                                                                                                                           | Pretrained |
| GPT-Neo      | 1.3B                                                                                                                          | Pretrained |
| GPT-J        | 6B                                                                                                                            | Pretrained |
| Incoder      | 6B                                                                                                                            | Pretrained |
| CodeParrot   | Small-python (110M), Small-multi(110M), 1.5B                                                                                   | Pretrained |
| CodeBERT     | CodeBERT-base, UnixCoder-base, CodeBERTa-small                                                                                 | Pretrained |


## Installation Guide

1. (Optional) Creating conda environment

```bash
conda create -n codetf python=3.8
conda activate codetf
```

2. Install from [PyPI](https://pypi.org/project/salesforce-codetf/):
```bash
pip install salesforce-codetf
```
    
3. Alternatively, build CodeTF from source:

```bash
git clone https://github.com/salesforce/CodeTF.git
cd CodeTF
pip install -e .
```

Additionally, to make sure the quantization feature works well, also install these dependencies:
```bash
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
```

For some models, such as [StarCoder](https://github.com/bigcode-project/starcoder), it is required to log in Huggingface. Please obtain the HuggingFace token and login:
```
huggingface-cli login
```

## Getting Started
### Inferencing Pipeline
    
Getting started with CodeTF is simple and quick with our model loading pipeline function ``load_model_pipeline()``. Here's an example showing how to load codet5+ model and perform inference on code generation task:
    
```python
from codetf.models import load_model_pipeline

code_generation_model = load_model_pipeline(model_name="codet5", task="pretrained",
            model_type="plus-770M-python", is_eval=True,
            load_in_8bit=True, load_in_4bit=False, weight_sharding=False)
            
result = code_generation_model.predict(["def print_hello_world():"])
print(result)
```
There are a few notable arguments that need to be considered:
-  ``model_name``: the name of the model, currently support ``codet5`` and ``causallm``. 
-  ``model_type``: type of model for each model name, e.g. ``base``, ``codegen-350M-mono``, ``j-6B``, etc.
-  ``load_in_8bit`` and ``load_in_4bit``: inherit the dynamic quantization feature from [Huggingface Quantization](https://huggingface.co/docs/transformers/main/main_classes/quantization).
-  ``weight_sharding``: our advance feature that leverages [HuggingFace Sharded Checkpoint](https://huggingface.co/docs/accelerate/v0.19.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch) to split a large model in several smaller shards in different GPUs. Please consider using this if you are dealing with large models.

### Model Zoo
You might want to view all of the supported models. To do this, you can use the ``model_zoo()``:
```python
from codetf.models import model_zoo
print(model_zoo)
# ============================================================================================================
# Architectures                  Types                           Tasks
# ============================================================================================================
# causallm                       codegen-350M-mono              pretrained
#                                codegen-350M-multi             pretrained
#                                codegen-350M-nl                pretrained
#                                codegen-2B-mono                pretrained
#                                codegen-2B-multi               pretrained
#                                codegen-2B-nl                  pretrained
#                                codegen-6B-mono                pretrained
#                                codegen-6B-nl                  pretrained
#                                codegen-6B-multi               pretrained
#                                starcoder-15.5B                pretrained
#                                gpt-neox-20B                   pretrained
#                                gpt-neo-1.3B                   pretrained
#                                gpt-j-6B                       pretrained
#                                incoder-6B                     pretrained
#                                codegen2-1B                    pretrained
#                                codegen2-3.7B                  pretrained
#                                codegen2-7B                    pretrained
#                                codegen2-16B                   pretrained
# codet5                         base-multi-sum                 pretrained
#                                base                           nl2code
#                                base                           refine
#                                base                           translate_cs_java
#                                base                           translate_java_cs
#                                base                           sum_python
#                                base                           sum_go
#                                base                           sum_php
#                                base                           sum_javascript
#                                base                           sum_java
#                                base                           sum_ruby
#                                base                           clone
#                                base                           defect
#                                plus-instruct-16B              pretrained
#                                plus-16B                       pretrained
#                                plus-6B                        pretrained
#                                plus-2B                        pretrained
#                                plus-770M-python               pretrained
#                                plus-770M                      pretrained
#                                plus-220M                      pretrained
# bert                           codebert-base                  pretrained
#                                unixcoder-base                 pretrained
#                                codeberta-small                pretrained
```

### Fine-Tuning Pipeline
Want to train a custom LLM for code? We've got you covered. Below is an example using the ``Seq2SeqTrainer`` to fine-tune a [CodeT5+ pretrained model](https://github.com/salesforce/CodeT5), along with our dataset utilities, make it easy to fine-tune your models using the CodeXGLUE dataset. Here's an example:
    
```python
from codetf.trainer.codet5_trainer import CodeT5Seq2SeqTrainer
from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
from codetf.models import load_model_pipeline
from codetf.performance.evaluation_metric import EvaluationMetric
from codetf.data_utility.base_dataset import CustomDataset

model_class = load_model_pipeline(model_name="codet5", task="pretrained",
            model_type="plus-220M", is_eval=True)

dataset = CodeXGLUEDataset(tokenizer=model_class.get_tokenizer())
train, test, validation = dataset.load(subset="text-to-code")

train_dataset= CustomDataset(train[0], train[1])
test_dataset= CustomDataset(test[0], test[1])
val_dataset= CustomDataset(validation[0], validation[1])

evaluator = EvaluationMetric(metric="bleu", tokenizer=model_class.tokenizer)

# peft can be in ["lora", "prefixtuning"]
trainer = CodeT5Seq2SeqTrainer(train_dataset=train_dataset, 
                                validation_dataset=val_dataset, 
                                peft="lora",
                                pretrained_model_or_path=model_class.get_model(),
                                tokenizer=model_class.tokenizer)
trainer.train()
```

Comparing to [this script from StarCoder](https://github.com/bigcode-project/starcoder/blob/main/finetune/finetune.py), which requires ~300 LOCs to fine-tune a model, we only need 14 LOCs to do the same !!!


### Evaluate on Well-Known Benchmarks
Planning to reproduce the results of well-known benchmarks like ``Human-Eval``, but struggling with not achieving the same numbers as reported in the original papers? Worried about the complicated evaluation process? Don't worry, we've got you covered with an intuitive, easy-to-use interface. Here's a sample snippet demonstrating how to evaluate Human Eval using pass@k (k=[1,10,100]) as the metric:
```python
from codetf.models import load_model_pipeline
from codetf.data_utility.human_eval_dataset import HumanEvalDataset
from codetf.performance.model_evaluator import ModelEvaluator

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_class = load_model_pipeline(model_name="causallm", task="pretrained",
            model_type="codegen-350M-mono", is_eval=True,
            load_in_8bit=True, weight_sharding=False)

dataset = HumanEvalDataset(tokenizer=model_class.get_tokenizer())
prompt_token_ids, prompt_attention_masks, references= dataset.load()

problems = TensorDataset(prompt_token_ids, prompt_attention_masks)

evaluator = ModelEvaluator(model_class)
avg_pass_at_k = evaluator.evaluate_pass_k(problems=problems, unit_tests=references)
print("Pass@k: ", avg_pass_at_k)
```

Comparing to [this script from HuggingFace](https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/human_eval.py), which requires ~230 LOCs to evaluate on pass@k, we only need 14 LOCs to do the same !!!

### Loading Preprocessed Data
CodeTF provides the Dataset utility for several well-known datasets, such as CodeXGLUE, Human Eval, MBPP, and APPS. The following is an example of how to load the CodeXGLUE dataset:  

```python
from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=True)
dataset = CodeXGLUEDataset(tokenizer=tokenizer)
train, test, validation = dataset.load(subset="text-to-code")
```

The ``train``, ``test``, ``validation`` are returned in form of [Pytorch tensor](https://pytorch.org/docs/stable/tensors.html) to provide the flexilbity for the users to wrap it into higher-lever wrapper for their own use cases.

### Code Utilities
In addition to providing utilities for LLMs, CodeTF also equips users with tools for effective source code manipulation. This is crucial in the code intelligence pipeline, where operations like parsing code into an Abstract Syntax Tree (AST) or extracting code attributes (such as function names or identifiers) are often required (CodeT5). These tasks can be challenging to execute, especially when setup and multi-language support is needed. Our code utility interface offers a streamlined solution, facilitating easy parsing and attribute extraction from code across 15+ languages.


#### AST Parser in Multiple Languages

CodeTF includes AST parsers compatible with numerous programming languages. Here's an example showcasing the parsing of Apex code into an AST:
```python
from codetf.code_utility.apex.apex_code_utility import ApexCodeUtility

apex_code_utility = ApexCodeUtility()

sample_code = """
    public class SampleClass {    
        public Integer myNumber;
        
        **
        * This is a method that returns the value of myNumber.
        * @return An integer value
        */
        public Integer getMyNumber() {
            // Return the current value of myNumber
            return this.myNumber;
        }
    }
"""
ast = apex_code_utility.parse(sample_code)

# This will print the tree-sitter AST object
print(ast)
```

Then you can traverse the tree using the interface from [py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter)
```
root_node = ast.root_node
assert root_node.type == 'module'
assert root_node.start_point == (1, 0)
assert root_node.end_point == (3, 13)
```

There are also other utilities for Java, Python, etc, that can perform the same operations. 

#### Extract Code Attributes

CodeTF provides an interface to easily extract code attributes. The following is a sample for extracting the function name of a Python function:

```python
code_attributes = apex_code_utility.get_code_attributes(sample_code)
print(code_attributes)
```

This will print:
``
{'class_names': ['AccountWithContacts'], 'method_names': ['getAccountsWithContacts'], 'comments': [], 'variable_names': ['acc', 'accounts', 'con', 'System', 'debug', 'Contacts', 'Id', 'Name', 'Account', 'Email', 'LastName']}
``

### Remove Comments
There are other existing utilities, such as removing comments from code:
```python
new_code_snippet = apex_code_utility.remove_comments(sample_code)
print(new_code_snippet)
```

This will print:
```java
public class SampleClass {    
        public Integer myNumber;
        public Integer getMyNumber() {
            return this.myNumber;
        }
    }
 ```

Note that this is an ongoing process, we will add more features to extract complicated code attributes in the future. More examples can be found [here](https://github.com/salesforce/CodeTF/tree/main/test_code_utilities).

## More Examples
You can find more examples for each use case:
- [Fine-tuning](https://github.com/salesforce/CodeTF/tree/main/test_trainer)
- [Inferencing](https://github.com/salesforce/CodeTF/tree/main/test_inference)
- [Model Evaluate](https://github.com/salesforce/CodeTF/tree/main/test_evaluation)
- [Code Utility](https://github.com/salesforce/CodeTF/tree/main/test_code_utilities)

## Notes
- CodeTF is designed to complement and enhance the capabilities of [HuggingFace Transformers](https://huggingface.co/docs/transformers/index), rather than replace it. It serves as a specialized layer specifically tailored for code intelligence tasks, such as fine-tuning language models with code-specific features and evaluating on well-known code intelligence benchmarks. If users require more customization, they are encouraged to write their own training code from scratch.
- CodeTF leverages the powerful functionality provided by [Accelerate](https://github.com/huggingface/accelerate) for both inference and training. With Accelerate, users do not need to manually manage GPUs or CPU devices for most operations, allowing for a streamlined and efficient workflow.

## Ethical and Responsible Use
CodeTF, while powerful, does not guarantee infallible code intelligence capabilities. Users may encounter inaccuracies or biases, possibly leading to misinterpretations or undesired behaviors. Risks include the generation of insecure code, propagation of poor coding practices, or inadvertent revelation of sensitive data. We strongly advise users to examine the pretrained models and system before practical adoption. CodeTF facilitates effective code analysis, prediction, and debugging, promoting reproducible research and development. We encourage its responsible use for enhancing software quality and developer productivity.

However, misuse can lead to unethical outcomes such as unauthorized code manipulation, privacy breaches, or insecure coding practices. Users should familiarize themselves with guidelines for responsible AI before using CodeTF. Our commitment is to continually refine the library by identifying and mitigating potential biases and inappropriate behaviors. Users should review the models and system before practical implementation, and contribute towards refining the library to ensure ethical usage.

## Technical Report and Citing CodeTF
You can find more details in our [technical report](https://arxiv.org/abs/2306.00029).

If you're using CodeTF in your research or applications, please cite using this BibTeX:
```bibtex
@misc{nghi2023codetf,
      title={CodeTF: A Transformer-based Library for CodeLLM & Code Intelligence}, 
      author={Nghi D. Q. Bui, Henry Le, Yue Wang, Akhilesh Deepak Gotmare, Junnan Li, Steven Hoi.},
      year={2023},
      eprint={2209.09019},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact us
If you have any questions, comments or suggestions, please do not hesitate to contact us at codetf@salesforce.com.

## License
[Apache License Version 2.0](LICENSE.txt)
