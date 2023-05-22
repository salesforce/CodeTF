import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
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