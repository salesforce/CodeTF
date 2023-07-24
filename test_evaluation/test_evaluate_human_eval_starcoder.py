import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from codetf.models import load_model_pipeline
from codetf.data_utility.human_eval_dataset import HumanEvalDataset
from codetf.performance.model_evaluator import ModelEvaluator
from torch.utils.data import TensorDataset
import os

os.environ["HF_ALLOW_CODE_EVAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4,5,6,7"

model_class = load_model_pipeline(model_name="causallm", task="pretrained",
            model_type="starcoder-15.5B", is_eval=True,
            load_in_8bit=True, weight_sharding=False)

dataset = HumanEvalDataset(tokenizer=model_class.get_tokenizer())
prompt_token_ids, prompt_attention_masks, references= dataset.load()

problems = TensorDataset(prompt_token_ids, prompt_attention_masks)

evaluator = ModelEvaluator(model_class)
avg_pass_at_k = evaluator.evaluate_pass_k(problems=problems, unit_tests=references, sequences_per_chunk=5, num_workers=5)
print("Pass@k: ", avg_pass_at_k)

