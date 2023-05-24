Model Serving Pipeline
################################################
Getting started with CodeTF is straightforward and swift with our model loading pipeline function load_model_pipeline(). Here's an example showing how to load CodeT5 models and perform inference on code translation and code summarization:

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

model_name: the name of the model, currently support codet5 and causal-lm.
model_type: type of model for each model name, e.g. base, codegen-350M-mono, j-6B, etc.
load_in_8bit: inherit the load_in_8bit feature from HuggingFace Quantization <https://huggingface.co/docs/transformers/main/main_classes/quantization>_.
weight_sharding: our advanced feature that leverages HuggingFace Sharded Checkpoint <https://huggingface.co/docs/accelerate/v0.19.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch>_ to split a large model into several smaller shards in different GPUs. Please consider using this if you are dealing with large models.