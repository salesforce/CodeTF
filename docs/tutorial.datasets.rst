Loading Preprocessed Data
################################################
CodeTF provides the Dataset utility for several well-known datasets, such as CodeXGLUE, Human Eval, MBPP, and APPS. The following is an example of how to load the CodeXGLUE dataset:

.. code-block:: python
	from codetf.data_utility.codexglue_dataset import CodeXGLUEDataset
	from transformers import RobertaTokenizer

	tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base", use_fast=True)
	dataset = CodeXGLUEDataset(tokenizer=tokenizer)
	train, test, validation = dataset.load(subset="text-to-code")

The train, test, and validation are returned in the form of a Pytorch tensor <https://pytorch.org/docs/stable/tensors.html>_ to provide flexibility for the users to wrap it into higher-level wrappers for their own use cases.
