from transformers import AutoTokenizer, BloomModel
import torch

# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
# tokenizer = AutoTokenizer.from_p

# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-560m")
model = BloomModel.from_pretrained("bigscience/bloom-560m")

with open("code_samples/sample_python.py", "r") as f:
    code = str(f.read())

prompt = f"Explain the logic of this code snippet: \n {code} \n Explanation: "
# code = 

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids
inputs = tokenizer(prompt, return_tensors="pt")

# gen_tokens = model.generate(
#     input_ids,
#     do_sample=True,
#     temperature=0.9,
#     max_length=100,
# )

outputs = model(**inputs)

# gen_text = tokenizer.batch_decode(outputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)