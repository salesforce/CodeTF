from codetf.code_utility.evaluator import Evaluator
from codetf.data_utility.human_eval_dataloader import HumanEvalDataLoader
from codetf.models import load_model


model_class = load_model(model_name="codet5", 
                model_type="base", task="nl2code", language="python", 
                is_eval=True, quantize=None)


dataloader = HumanEvalDataLoader(tokenizer=model_class.tokenizer)
prompts, solutions, test_cases = dataloader.load_data()

eval = Evaluator()

# predict_raw() is for raw input, predict() is for post-processing to tensors
predictions = model_class.predict_raw(prompts)

results = eval.evaluate_pass_k(predictions, test_cases, k=1)

print(results)






