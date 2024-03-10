from langchain.chains import LLMChain
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate


class FinancialAnalysisChainOneShot(LLMChain):
    description = "One shot agent which takes a question and data in a context to provide an answer"

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
            template=analysis_prompt, input_variables=["data", "question"]
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose)


class FinancialAnalysisChainWithHistory(LLMChain):
    description = (
        "The agent is working in 3 stages,"
        "first by splitting a questions into several smaller ones"
        "and keeping the history of conversation through the conversation."
    )

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        analysis_prompt = """You work as financial analyst helping your company understand financial report.
When asked about ratio, proportion, or change, you must provide your answer using percentage value.
{context}
Conversation history:
{conversation_history}
        """

        prompt = PromptTemplate(
            template=analysis_prompt, input_variables=["data", "question"]
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose)
