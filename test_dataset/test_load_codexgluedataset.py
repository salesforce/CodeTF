import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from codetf.data_utility.codexglue_dataloader import CodeXGLUEDataLoader
from transformers import RobertaTokenizer

def main():
    tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
    
    dataloader = CodeXGLUEDataLoader(tokenizer=tokenizer)
    train_dataset, test_dataset, val_dataset = dataloader.load_codexglue_code_to_text_dataset()
    print(train_dataset[1])


if __name__ == "__main__":
    main()