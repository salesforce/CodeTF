Evaluate on Well-Known Benchmarks
If you're planning to reproduce the results of well-known benchmarks like Human-Eval, but are struggling with not achieving the same numbers as reported in the original papers and are worried about the complicated evaluation process, we've got you covered with an intuitive, easy-to-use interface. Here's a sample snippet demonstrating how to evaluate Human Eval using pass@k (k=[1,10,100]) as the metric:

.. code-block:: python
    from codetf.models import load_model_pipeline
    from codetf.data_utility.human_eval_dataset import HumanEvalDataset
    from codetf.performance.model_evaluator import ModelEvaluator
    import os
    from torch.utils.data import TensorDataset

    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    model_class = load_model_pipeline(model_name="causal-lm", task="pretrained",
                model_type="codegen-350M-mono", is_eval=True,
                load_in_8bit=True, weight_sharding=False)

    dataset = HumanEvalDataset(tokenizer=model_class.get_tokenizer())
    prompt_token_ids, prompt_attention_masks, references = dataset.load()

    problems = TensorDataset(prompt_token_ids, prompt_attention_masks)

    evaluator = ModelEvaluator(model_class)
    avg_pass_at_k = evaluator.evaluate_pass_k(problems=problems, unit_tests=references)
    print("Pass@k: ", avg_pass_at_k)
