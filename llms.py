from abc import ABC


class LLMStrategy(ABC):
    def get_llm(self, name):
        pass
