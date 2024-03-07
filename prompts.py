from abc import ABC
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)


class PromptingStrategy(ABC):
    def get_prompt(self):
        pass




class PromptWithFewShots(PromptingStrategy):
    def get_prompt(self):
        examples = [
            {
                "input": "what is the percentage change in revenue from year 2008 to 2009 given in 2008 it was 100 and in 2009 110",
                "output": "10%",
            },
            {
                "input": "what portion of total invoices are due given 35 out of 100 are late",
                "output": "35%",
            },
        ]

        example_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{input}"),
                ("ai", "{output}"),
            ]
        )
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=examples,
        )

        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Your task is to provide an answer to a user's question quoted in triple backticks below. 
                    In order to do it, first, retrieve required data from the API using a query string parameter of id={document_id}
                    """,
                ),
                few_shot_prompt,
                ("human", "I need your help with the following question."),
                ("human", "```{question}```"),
            ]
        )


class PromptSimple(PromptingStrategy):

    def get_prompt(self):

        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Your task is to provide an answer to a user's question."
                    "In order to do it, first, retrieve required data from the API using a query string parameter of id={document_id}",
                ),
                ("human", "I need your help with the following question."),
                ("human", "{question}"),
            ]
        )


class PromptAWithAPIDocs(PromptingStrategy):

    def get_prompt(self):

        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Your task is to provide an answer to a user's question quoted in triple backticks.
                    In order to do it, first, retrieve required data from the API using a query string parameter of id={document_id}
                    
                    API response would the following fields:
                    Name    Description
                    table   tabular view of the financial data
                    pre_text    text that appears before the table in a document, describing what is a a table
                    post_text   a text that appears after the table in a document, providing analysis of the data in a table
                    """,
                ),
                ("human", "```{question}```"),
            ]
        )


class PromptStepByStepInstructions(PromptingStrategy):

    def get_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Your task is to analyse user's question and provide a step-by-step instructions on how to resolve it.
                    In order to do it, first, retrieve required data from the API using a query string parameter of id={document_id}
                    """,
                ),
                ("human", "```{question}```"),
            ]
        )


class PromptGeneratePseudoCode(PromptingStrategy):

    def get_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Your task is to 
                    1) retrieve required data from the API using a query string parameter of id={document_id}
                    2) analyse user's question quoted in triple backticks below
                    3) write pseudo code in Python to answer user's question
                    """,
                ),
                ("human", "```{question}```"),
            ]
        )


class PromptExecutePseudoCode(PromptingStrategy):
    def get_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    Your task is to 
                    1) retrieve required data from the API using a query string parameter of id={document_id}
                    2) follow pseudo code provided below ste-by-step 
                    3) answer the user question quoted in triple backticks below.
                    """,
                ),
                ("ai", "{pseudo_code}"),
                ("human", "```{question}```")
            ]
        )
