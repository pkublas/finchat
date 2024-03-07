from abc import ABC
from langchain_openai import OpenAI


class LLMStrategy(ABC):
    def get_llm(self, temperature=0):
        pass


class OpenAIStrategy(LLMStrategy):
    def get_llm(self, temperature=0):
        return OpenAI(temperature=temperature)
