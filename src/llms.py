from abc import ABC
from langchain_openai import ChatOpenAI
from langchain_community.llms import Bedrock


class LLMStrategy(ABC):
    def get_llm(self, temperature=0):
        pass


class OpenAIStrategy(LLMStrategy):

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "gpt-4")
        self.temperature = kwargs.get("temperature", 0)

    def get_llm(self):
        return ChatOpenAI(temperature=self.temperature, model=self.model_name)


class BedrockStrategy(LLMStrategy):

    def __init__(self, **kwargs):
        self.model_name = kwargs.get("model_name", "amazon.titan-text-express-v1")
        self.credentials_profile_name = kwargs.get(
            "credentials_profile_name", "default"
        )

    def get_llm(self):
        return Bedrock(
            credentials_profile_name=self.credentials_profile_name,
            model_id=self.model_name,
        )
