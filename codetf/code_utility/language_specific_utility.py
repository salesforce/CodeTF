from .base_utility import BaseUtility
from abc import ABC, abstractmethod

class LanguageSpecificUtility(BaseUtility):
    def __init__(self, language: str):
        super(LanguageSpecificUtility, self).__init__(language)