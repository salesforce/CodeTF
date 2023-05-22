import re
from transformers import StoppingCriteria

EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]

# def remove_last_block(string):
#     """Remove the last block of the code containing EOF_STRINGS"""
#     string_list = re.split("(%s)" % "|".join(EOF_STRINGS), string)
#     # last string should be ""
#     return "".join(string_list[:-2])

def remove_last_block(string):
    """Remove the last block of the code if it's incomplete"""
    lines = string.split('\n')
    
    # Detect the last complete block
    last_complete_block = -1
    for i in reversed(range(len(lines))):
        stripped_line = lines[i].strip()
        if stripped_line.endswith(':') or stripped_line.endswith(')') or stripped_line.endswith(']') or stripped_line.endswith('}') or '=' in stripped_line or 'return' in stripped_line:
            last_complete_block = i
            break

    # If there is an incomplete block at the end, remove it
    if last_complete_block != len(lines) - 1:
        return '\n'.join(lines[:last_complete_block+1])

    return string

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer
      

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(input_ids[:, self.start_length :])
        done = []
        for decoded_generation in decoded_generations:
            done.append(any([stop_string in decoded_generation for stop_string in self.eof_strings]))
        return all(done)


