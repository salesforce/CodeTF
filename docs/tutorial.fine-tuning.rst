Fine-Tuning Custom Model Using Our Trainer
################################################
If you want to train a custom LLM for code, we've got you covered. Below is an example using the CausalLMTrainer, along with our dataset utilities, that makes it easy to fine-tune your models using the CodeXGLUE dataset. Here's an example:

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

Comparing to this script from StarCoder <https://github.com/bigcode-project/starcoder/blob/main/finetune/finetune.py>_, which requires approximately 300 lines of code to fine-tune a model, we only need 14 lines of code to do the same!