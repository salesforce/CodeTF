import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = load_model(model_name="codet5", task="refine",
                model_type="base",
                is_eval=True, quantize="int8", quantize_algo="bitsandbyte")

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
refined = model.predict([code_snippets])

print(refined)