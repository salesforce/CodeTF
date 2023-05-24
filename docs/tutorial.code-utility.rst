Code Utilities
################################################
In addition to providing utilities for LLMs, CodeTF also equips users with tools for effective source code manipulation. This is crucial in the code intelligence pipeline, where operations like parsing code into an Abstract Syntax Tree (AST) or extracting code attributes (such as function names or identifiers) are often required (CodeT5). These tasks can be challenging to execute, especially when setup and multi-language support is needed. Our code utility interface offers a streamlined solution, facilitating easy parsing and attribute extraction from code across 15+ languages.


CodeTF includes AST parsers compatible with numerous programming languages. Here's an example showcasing the parsing of Apex code into an AST:

.. code-block:: python

    from codetf.code_utility.apex.apex_code_utility import ApexCodeUtility

    apex_code_utility = ApexCodeUtility()

    sample_code = """
        public class SampleClass {    
            public Integer myNumber;
            
            **
            * This is a method that returns the value of myNumber.
            * @return An integer value
            */
            public Integer getMyNumber() {
                // Return the current value of myNumber
                return this.myNumber;
            }
        }
    """
    ast = apex_code_utility.parse(sample_code)

    # This will print the tree-sitter AST object
    print(ast)

Then you can traverse the tree using the interface from `py-tree-sitter <https://github.com/tree-sitter/py-tree-sitter>`_

.. code-block:: python

    root_node = ast.root_node
    assert root_node.type == 'module'
    assert root_node.start_point == (1, 0)
    assert root_node.end_point == (3, 13)

There are also other utilities for Java, Python, etc, that can perform the same operations.

Extract Code Attributes
~~~~~~~~~~~~~~~~~~~~~~~

CodeTF provides an interface to easily extract code attributes. The following is a sample for extracting the function name of a Python function:

.. code-block:: python

    code_attributes = apex_code_utility.get_code_attributes(sample_code)
    print(code_attributes)

This will print:
``
{'class_names': ['AccountWithContacts'], 'method_names': ['getAccountsWithContacts'], 'comments': [], 'variable_names': ['acc', 'accounts', 'con', 'System', 'debug', 'Contacts', 'Id', 'Name', 'Account', 'Email', 'LastName']}
``

Remove Comments
~~~~~~~~~~~~~~~

There are other existing utilities, such as removing comments from code:

.. code-block:: python

    new_code_snippet = apex_code_utility.remove_comments(sample_code)
    print(new_code_snippet)

This will print:
```
public class SampleClass {    
        public Integer myNumber;
        public Integer getMyNumber() {
            // Return the current value of myNumber
            return this.myNumber;
        }
    }
``

Note that this is an ongoing process, we will add more features to extract complicated code attributes in the future. More examples can be found `here <https://github.com/salesforce/CodeTF/tree/main/test_code_utilities>`.
