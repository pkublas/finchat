from langchain.chains import LLMChain
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate


class FinancialAnalysisChainOneShot(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        analysis_prompt = """You work as financial analyst helping your company understand financial report.
        Answer the question quoted in triple backticks below given the data available between `===`.
        ```{question}```
        ===
        {data}
        ===
        """

        prompt = PromptTemplate(
            template=analysis_prompt, input_variables=["question", "data"]
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose)


class FinancialAnalysisChainWithHistory(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        analysis_prompt = """You work as financial analyst helping your company understand financial report.
When asked about ratio, proportion, or change, you must provide your answer using percentage value.
{context}
Conversation history:
{conversation_history}
        """

        prompt = PromptTemplate(template=analysis_prompt, input_variables=["context", "conversation_history"])

        return cls(prompt=prompt, llm=llm, verbose=verbose)


class FinancialAnalysisChainWithCode(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        analysis_prompt = """You are Python developer working with financial data.
You must validate your answers using Python's numexpr expression.
When asked about ratio, proportion, or change, you must provide your answer using percentage value.
{context}
        """

        prompt = PromptTemplate(template=analysis_prompt, input_variables=["context"])

        return cls(prompt=prompt, llm=llm, verbose=verbose)
