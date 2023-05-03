from fastapi import FastAPI
import torch
from codetf.models import load_model

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(name="codet5_nl2code", model_type="large", is_eval=True, device=device)

async def get_global_object():
    return model

@app.get("/code_summarization")
def summarize_code():
    
    prompts = "# check if a string is in valid format"
    code_snippets = model.predict([prompts])

    return code_snippets