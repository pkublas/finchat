from abc import ABC

from langchain.chains.base import Chain
from langchain_core.language_models import BaseLLM

from llms import LLMStrategy


class AgentFinancialAnalyst:

    def __init__(self, llm: BaseLLM, chain: Chain):
        self.llm = llm.get_llm()
        self.chain = chain.from_llm(self.llm)

    def restart(self):
        pass

    def start(self, input):
        answer = self.chain.invoke(
            input={
                "question": input.get("question"),
                "data": input.get("data"),
            }
        )
        print(answer)
        return answer


class AgentFinancialAnalystWithHistory:

    def __init__(self, llm: LLMStrategy, chain: Chain):
        self.llm = llm.get_llm()
        self.chain = chain.from_llm(self.llm)
        self.conversation_history = []
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

    def start(self, input):

        print(input)
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
