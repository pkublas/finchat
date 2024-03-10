from langchain.chains import LLMMathChain
from langchain_core.language_models import BaseLLM

from chains import FinancialAnalysisChainWithHistory, FinancialAnalysisChainOneShot, FinancialAnalysisChainWithCode
from llms import LLMStrategy


class AgentFinancialAnalystSimple:

    def __init__(self, llm: BaseLLM, verbose=True):
        self.llm = llm.get_llm()
        self.chain = FinancialAnalysisChainOneShot.from_llm(self.llm, verbose=verbose)
        self.description = "One shot chain which takes a question and data in a context to provide an answer"

    def restart(self):
        pass

    def start(self, input):
        print(f"input={input}")
        answer = self.chain.invoke(
            input={
                "data": input.get("data"),
                "question": input.get("question"),
            }
        )
        print(answer)
        return answer


class AgentFinancialAnalystWithHistory:

    def __init__(self, llm: LLMStrategy, verbose=True):
        self.llm = llm.get_llm()
        self.chain = FinancialAnalysisChainWithHistory.from_llm(self.llm, verbose=verbose)
        self.conversation_history = []
        self.description = (
            "The agent is working in 3 stages, "
            "first by splitting a questions into several smaller ones and "
            "keeping the history of conversation through the conversation."
        )
        self.stages = [
            """Provide a set of several question which when resolved would lead to 
to the answer of a question quoted in triple backticks based on data available between ===.
```{question}```
===
{data}
===
            """,
            """Answer each question quoted in triple backticks
given the data available between === in a conversation history.
```{question}```.""",
            """Given the history of the conversation, answer the final question quoted in triple backticks 
given intermediary answers in conversation history.
```{question}```
            """,
        ]

    @property
    def model_name(self):
        return self.llm.name

    @property
    def chain_description(self):
        return self.chain.description

    def restart(self):
        self.conversation_history = []

    # TODO repeated code, add method
    def start(self, input):

        print(f"input={input}")
        context1 = self.stages[0].format(
            **{"question": input.get("question"), "data": input.get("data")}
        )
        answer1 = self.chain.invoke(
            input={"context": context1, "conversation_history": None}
        )
        self.conversation_history.append(("human", f"{context1}"))
        self.conversation_history.append(("ai", f"{answer1.get('text')}"))
        print(f"answer1={answer1}")

        context2 = self.stages[1].format(**{"question": answer1.get("text")})
        answer2 = self.chain.invoke(
            input={
                "context": context2,
                "conversation_history": "\n".join(
                    f"{item[0]}:{item[1]}" for item in self.conversation_history
                ),
            }
        )
        print(f"answer2={answer2}")
        self.conversation_history.append(("human", f"{context2}"))
        self.conversation_history.append(("ai", f"{answer2.get('text')}"))

        context3 = self.stages[2].format(**{"question": input.get("question")})
        answer3 = self.chain.invoke(
            input={
                "context": context3,
                "conversation_history": "\n".join(
                    f"{item[0]}:{item[1]}" for item in self.conversation_history
                ),
            }
        )
        print(f"answer3={answer3}")
        return answer3.get("text")


class AgentFinancialAnalystDeveloper:

    def __init__(self, llm: BaseLLM, verbose=True):
        self.llm = llm.get_llm()
        self.chain = FinancialAnalysisChainWithCode.from_llm(self.llm, verbose=verbose)
        self.math_chain = LLMMathChain.from_llm(llm=self.llm)
        self.description = "The agent working in stages to generate Python's numexpr expression to produce answers."
        self.stages = [
            """Your task is to translate the question quoted in triple backticks below into Python's numexpr expression
using descriptive variable names.
Question: {question}
            """,
            """
            Update provided Python's numexpr expression quoted in triple backticks to include required values from data between === below.
Expression: {expression}
===
{data}
===
            """,
            """
            Execute Python's numexpr expression quoted in triple backticks to produce an answer.
Expression: {expression}
            """,
        ]

    def restart(self):
        pass

    # TODO repeated code, add method
    def start(self, input):

        print(f"input={input}")
        context1 = self.stages[0].format(**{"question": input.get("question")})
        answer1 = self.chain.invoke(input={"context": context1})

        print(f"answer1={answer1}")

        context2 = self.stages[1].format(
            **{"expression": answer1.get("text"), "data": input.get("data")}
        )
        answer2 = self.chain.invoke(input={"context": context2})
        print(f"answer2={answer2}")

        context3 = self.stages[2].format(**{"expression": answer2.get("text")})
        answer3 = self.math_chain.invoke(input={"question": context3})
        print(f"answer3={answer3}")
        return answer3.get("answer")
