import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
from codetf.code_utility.apex.apex_code_utility import ApexCodeUtility

apex_code_utility = ApexCodeUtility()

sample_code = """
    public class SampleClass {    
        public Integer myNumber;
        public Integer getMyNumber() {
            // Return the current value of myNumber
            return this.myNumber;
        }
    }
"""

ast = apex_code_utility.parse(sample_code)

# This will print the tree-sitter AST object
print(ast)