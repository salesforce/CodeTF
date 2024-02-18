from .ast_parser import ASTParser
import lizard
import re

class BaseUtility():

    def __init__(self, language: str):
        self.language = language
        self.parser = ASTParser(language) 
    
    def parse(self, code_snippet):
        tree = self.parser.parse(bytes(code_snippet, "utf8"))
        return tree

    def remove_comments(self, code_snippet):
        if self.language == "python":
            commentFilter = pyparsing.pythonStyleComment.suppress()
        elif self.language == "c":
            commentFilter = pyparsing.cStyleComment.suppress()
        elif self.language == "cpp":
            commentFilter = pyparsing.cppStyleComment.suppress()
        else:
            commentFilter = pyparsing.javaStyleComment.suppress()
        
        code_snippet = commentFilter.transformString(code_snippet)
        return code_snippet


    def get_method_name(self, function_text):
        first_line = function_text.splitlines()[0]
        match = re.search(r"(?<=\s)\w+(?=\()", first_line)
        function_name = None
        if match:
            function_name = match.group()
        
        return function_name

