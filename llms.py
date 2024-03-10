from abc import ABC
from langchain_openai import ChatOpenAI
from langchain_community.llms import Bedrock


class LLMStrategy(ABC):
    def get_llm(self, temperature=0):
        pass


class OpenAIStrategy(LLMStrategy):
    name = "OpenAI=gpt-4"

    @classmethod
    def get_llm(cls, temperature=0, model="gpt-4"):
        return ChatOpenAI(temperature=temperature, model=model)


class BedrockStrategy(LLMStrategy):
    name = "Bedrock-amazon.titan-text-express-v1"

    @classmethod
    def get_llm(cls, temperature=0, model="amazon.titan-text-express-v1"):
        return Bedrock(credentials_profile_name="default", model_id=model)
