from langchain.chains import APIChain
from langchain.chains.api import open_meteo_docs
from langchain_openai import OpenAI
import dotenv

dotenv.load_dotenv()
llm = OpenAI(temperature=0)

finchat_api_docs = """
BASE URL: http://127.0.0.1:8001

API Documentation

API provides financial data from variety of documents at /table.

Query string parameters:
Name    Required    Description
id  yes identifies a document where to load a data from

API response includes the following fields:
Name    Description
table   tabular view of the financial data"
pre_text    text that appears before the table in a document,usually introducing what's in this table"
post_text   a text that appears after the table in a document, usually providing analyses of the data in a table

"""


def get_weather(location):
    chain = APIChain.from_llm_and_api_docs(
        llm,
        open_meteo_docs.OPEN_METEO_DOCS,
        verbose=True,
        limit_to_domains=["https://api.open-meteo.com/"],
    )
    chain.invoke(f"What is the weather like right now in {location}?")


def get_answer(document_id, question):
    chain = APIChain.from_llm_and_api_docs(
        llm,
        finchat_api_docs,
        verbose=True,
        limit_to_domains=["http://127.0.0.1:8001/"],
    )

    prompt = (
        f"Provide an answer (as a decimal number) to user's question quoted in triple backticks below"
        f"about a financial document they are working on."
        f"Retrieve that document's data via API using a single query string parameter of id={document_id}."
        f"```{question}```"
    )

    return chain.invoke(prompt)


if __name__ == "__main__":
    # document_id = "Single_JKHY_2009_page_28.pdf-3"
    # question = """
    #     what was the percentage change in the net cash from operating activities from 2008 to 2009?
    # """

    document_id = "Single_HIG_2004_page_122.pdf-2"
    question = """
        what portion of total obligations are due within the next 3 years?
    """
    answer = get_answer(document_id=document_id, question=question)
    print(answer.get("output"))
