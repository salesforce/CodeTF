Model Serving Pipeline
----------------------

Getting started with CodeTF is simple and quick with our model loading pipeline function ``load_model_pipeline()``. Here's an example showing how to load codet5 models and perform inference on code translation and code summarization:

.. code-block:: python

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

There are a few notable arguments that need to be considered:

- ``model_name``: the name of the model, currently support ``codet5`` and ``causal-lm``. 
- ``model_type``: type of model for each model name, e.g. ``base``, ``codegen-350M-mono``, ``j-6B``, etc.
- ``load_in_8bit``: inherit the ``load_in_8bit" feature from `Huggingface Quantization <https://huggingface.co/docs/transformers/main/main_classes/quantization>`_.
- ``weight_sharding``: our advance feature that leverate `HuggingFace Sharded Checkpoint <https://huggingface.co/docs/accelerate/v0.19.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch>`_ to split a large model in several smaller shards in different GPUs. Please consider using this if you are dealing with large models.

Fine-Tuning Custom Model Using Our Trainer
------------------------------------------

Want to train a custom LLM for code? We've got you covered. Below is an example using the ``CausalLMTrainer``, along with our dataset utilities, make it easy to fine-tune your models using the CodeXGLUE dataset. Here's an example:

.. code-block:: python

    from codetf.trainer.causal_lm_trainer import CausalLMTrainer
    from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
    from codetf.models import load_model_pipeline
    from codetf.performance.evaluate import EvaluationMetric

    model_class = load_model_pipeline(model_name="causal-lm", task="pretrained",
                    model_type="starcoder-15.5B", is_eval=False,
                    load_in_8bit=False, weight_sharding=False)


    dataloader = CodeXGLUEDataset(tokenizer=model_class.get_tokenizer())
    train_dataset, test_dataset, val_dataset = dataloader.load(subset="text-to-code")

    evaluator = EvaluationMetric(metric="bleu", tokenizer=model_class.tokenizer)

    # peft can be in ["lora", "prefixtuning"]
    trainer = CausalLMTrainer(train_dataset=train_dataset, 
                            validation_dataset=val_dataset, 
                            peft=None,
                            pretrained_model_or_path=model_class.get_model(),
                            tokenizer=model_class.get_tokenizer())
    trainer.train()
    # trainer.evaluate(test_dataset=test_dataset)

Comparing to `this script from StarCoder <https://github.com/bigcode-project/starcoder/blob/main/finetune/finetune.py>`_, which requires ~300 LOCs to fine-tune a model, we only need 14 LOCs to do the same !!!

Evaluate on Well-Known Benchmarks
---------------------------------

Planning to reproduce the results of well-known benchmarks like ``Human-Eval``, but struggling with not achieving the same numbers as reported in the original papers? Worried about the complicated evaluation process? Don't worry, we've got you covered with an intuitive, easy-to-use interface. Here's a sample snippet demonstrating how to evaluate Human Eval using pass@k (k=[1,10,100]) as the metric:

.. code-block:: python

    from codetf.models import load_model_pipeline
    from codetf.data_utility.human_eval_dataset import HumanEvalDataset
    from codetf.performance.model_evaluator import ModelEvaluator

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_class = load_model_pipeline(model_name="causal-lm", task="pretrained",
                model_type="codegen-350M-mono", is_eval=True,
                load_in_8bit=True, weight_sharding=False)

    dataset = HumanEvalDataset(tokenizer=model_class.get_tokenizer())
    prompt_token_ids, prompt_attention_masks, references= dataset.load()

    problems = TensorDataset(prompt_token_ids, prompt_attention_masks)

    evaluator = ModelEvaluator(model_class)
    avg_pass_at_k = evaluator.evaluate_pass_k(problems=problems, unit_tests=references)
    print("Pass@k: ", avg_pass_at_k)

Comparing to `this script from HuggingFace <https://github.com/huggingface/transformers/blob/main/examples/research_projects/codeparrot/scripts/human_eval.py>`_, which requires ~230 LOCs to evaluate on pass@k, we only need 14 LOCs to do the same !!!

Loading Preprocessed Data
-------------------------

CodeTF provides the Dataset utility for several well-known datasets, such as CodeXGLUE, Human Eval, MBPP, and APPS. The following is an example of how to load the CodeXGLUE dataset:

.. code-block:: python

    from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
    from transformers import RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=True)
    dataset = CodeXGLUEDataset(tokenizer=tokenizer)
    train, test, validation = dataset.load(subset="text-to-code")

The ``train``, ``test``, ``validation`` are returned in form of `Pytorch tensor <https://pytorch.org/docs/stable/tensors.html>`_ to provide the flexilbity for the users to wrap it into higher-lever wrapper for their own use cases.

Code Utilities
--------------

In addition to providing utilities for LLMs, CodeTF also equips users with tools for effective source code manipulation. This is crucial in the code intelligence pipeline, where operations like parsing code into an Abstract Syntax Tree (AST) or extracting code attributes (such as function names or identifiers) are often required (CodeT5). These tasks can be challenging to execute, especially when setup and multi-language support is needed. Our code utility interface offers a streamlined solution, facilitating easy parsing and attribute extraction from code across 15+ languages.

AST Parser in Multiple Languages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CodeTF includes AST parsers compatible with numerous programming languages. Here's an example showcasing the parsing of Apex code into an AST:

.. code-block:: python

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

Then you can traverse the tree using the interface from `py-tree-sitter <https://github.com/tree-sitter/py-tree-sitter>`_
    ::
        root_node = ast.root_node
        assert root_node.type == 'module'
        assert root_node.start_point == (1, 0)
        assert root_node.end_point == (3, 13)

There are also other utilities for Java, Python, etc, that can perform the same operations.

Extract Code Attributes
^^^^^^^^^^^^^^^^^^^^^^^

CodeTF provides an interface to easily extract code attributes. The following is a sample for extracting the function name of a Python function:

.. code-block:: python

    code_attributes = apex_code_utility.get_code_attributes(sample_code)
    print(code_attributes)

This will print:
    ::
        {'class_names': ['AccountWithContacts'], 'method_names': ['getAccountsWithContacts'], 'comments': [], 'variable_names': ['acc', 'accounts', 'con', 'System', 'debug', 'Contacts', 'Id', 'Name', 'Account', 'Email', 'LastName']}

Remove Comments
^^^^^^^^^^^^^^^

There are other existing utilities, such as removing comments from code:

.. code-block:: python

    new_code_snippet = apex_code_utility.remove_comments(sample_code)
    print(new_code_snippet)

This will print:
    ::
        public class SampleClass {    
            public Integer myNumber;
            public Integer getMyNumber() {
                // Return the current value of myNumber
                return this.myNumber;
            }
        }
"""
