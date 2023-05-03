import os
import re
import urllib
import fnmatch
import platform
from tqdm import tqdm
from pathlib import Path
from pprint import pprint
import torch
from guesslang import Guess
from .metadata import NAME2NET, AVAILABLE_MODELS
from tree_sitter import Parser, Language
import logging
import copy


class CodeTFRunner(object):
    PARSER = Parser()
    def __init__(self, task: str):
        self.accepted_languages = ['java', 'python', 'ruby', 'javascript', 'php', 'go', 'c_sharp']

        self.guess_lang = Guess()
        self.runner_class = self._init_model_checkpoint(task)

        if not os.path.exists(self._pretrained_model_path):
            pretrained_model_url = NAME2NET[self.pretrained_model_name]['url']
            download_url(pretrained_model_url, self._pretrained_model_path)

        self.net = NAME2NET[self.pretrained_model_name]['net'].load_checkpoint(self._pretrained_model_path)
        
        if torch.cuda.is_available():
            print("Using cuda....")
            self.net.cuda()

    def _init_class(self, task):
        assert self.pretrained_model_name in AVAILABLE_MODELS
        home = str(Path.home())
        filename = NAME2NET[self.pretrained_model_name]['filename']
        self._pretrained_model_path = os.path.join(home, '.code2text', 'model_checkpoints', filename)

    def execute(self, code_snippet):
        input_for_net = [' '.join(code_snippet.strip().split()).replace('\n', ' ')]

        predictions = self.net.predict(input_for_net, num_samples=num_samples, decoding=decoding)
        arbitrary_snippets = []
        arbitrary_snippet = {}
        arbitrary_snippet["commentData"] = predictions[0]
        arbitrary_snippet["blockType"] = "IN_LINE"
        arbitrary_snippet["line"] = 0

        arbitrary_snippets.append(arbitrary_snippet)
        return arbitrary_snippets

    def summarize(self, code_snippet: str, num_samples=1, decoding="beam_search", type="arbitrary", 
                language="auto-detect", intelligent_detection=True):

        if check_if_string_is_none(code_snippet):
            return None

        if language != "auto-detect" and language:
            self.set_language(language)
        else:
            language = self.guess_lang.language_name(code_snippet).lower()
            if language == None or language not in self.accepted_languages:
                self.LOGGER.info("Cannot detect language, using java as the default")
                language = "java"
            else:
                self.LOGGER.info("Detected language of snippet : " + language)
                print(language)
                if language not in self.accepted_languages:
                    language = self.language
            self.set_language(language)


        parser_class = self.select_parser_class(self.language)
        output_dict = {}

        ast = self.PARSER.parse(code_snippet.encode()) 

      
        results = self.summarize_arbitrary(code_snippet, num_samples=num_samples, decoding=decoding)
                  
        output_dict["comments"] = results
        output_dict["language"] = language
        
        return output_dict

  
    def set_language(self, language):
        assert language in ['java', 'python', 'ruby', 'javascript', 'php', 'go', 'c_sharp']
        self.language = language
        self.PARSER.set_language(get_language(language))

    
    def select_parser_class(self, language):
        parser_class = language_parser_mapper[language]
        return parser_class
