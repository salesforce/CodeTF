import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute().parent))
# sys.path.append(str(Path(__file__).resolve().parents[2]))
from codetf.code_utility.apex.apex_code_utility import ApexCodeUtility

apex_code_utility = ApexCodeUtility()

sample_code = """
    /**
    * This is a simple Apex class named 'SampleClass'.
    * It demonstrates a basic use of Apex with Salesforce.com.
    */
    public class SampleClass {

        // This is a public integer variable
        public Integer myNumber;

        // This is a constructor method for SampleClass
        public SampleClass() {
            // Initializing myNumber with a value
            this.myNumber = 5;
        }

        /**
        * This is a method that increments the value of myNumber.
        * It takes no parameters and returns no value.
        */
        public void incrementNumber() {
            // Increment the value of myNumber by 1
            this.myNumber++;
        }

        /**
        * This is a method that returns the value of myNumber.
        * @return An integer value
        */
        public Integer getMyNumber() {
            // Return the current value of myNumber
            return this.myNumber;
        }
    }
"""


# for sample_code in sample_codes:
new_code_snippet = apex_code_utility.remove_comments(sample_code)
print(new_code_snippet)


