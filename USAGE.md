## Sample usage for code summarization 
```python
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

summarization_model = load_model(name="codet5_summarization", model_type="base", is_eval=True, device=device)
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
summarizations = model.predict([summarization_model])
print(summarizations)
```

## Sample usage for code translation
```python
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

translation_model = load_model(name="codet5_translation", model_type="base", is_eval=True, device=device)
code_snippet = """
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
translated_code = translation_model.predict([code_snippet])
print(translated_code)
```


## Sample usage for code generation
```python
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generation_model = load_model(name="codet5_generation", model_type="base", is_eval=True, device=device)
prompt = "check if a string is in email format"
code_snippet = generation_model.predict([prompt])
print(code_snippet)
```

## Sample usage for code refinement
```python
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

refinement_model = load_model(name="codet5_refinement", model_type="base", is_eval=True, device=device)
code_snippets = """
      private static List<Field> getAllFields(Class clazz) {
        ArrayList<Field> allFields = new ArrayList<Field>();
        Class current = clazz;
        while (current != null) {
            allFields.addAll(Arrays.asList(current.getDeclaredFields()));
            current = current.getSuperclass();
        }
        return allFields;
    }
"""
refined_code = refinement_model.predict([code_snippets])
print(refined_code)
```

## Sample usage for anti-pattern detection
```python
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append("../")
import torch
from codetf.models import load_model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

prom_model = load_model(name="codet5_apexprom", model_type="base", is_eval=True, device=device)
code_snippets = """
     public static testMethod void setDataToSendDataAuthorizationMail7() {
        List<TPA_Request__c> ObjTpaList = [select id from TPA_Request__c];
        TGRH_TPARequest controller = new TGRH_TPARequest();
        controller.onBeforeDelete(ObjTpaList);
    }
"""
report = refinement_model.predict([code_snippets])
print(report)
```
