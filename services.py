from langchain.chains import APIChain, LLMChain
from langchain.chains.api import open_meteo_docs
from langchain_openai import OpenAI

import dotenv
import json

from langchain_core.prompts import PromptTemplate

from llms import LLMStrategy
from prompts import PromptingStrategy

LOCAL_API_DOCS = """
BASE URL: http://127.0.0.1:8001

API Documentation

API provides financial data from variety of documents at /table.

Query string parameters:
Name    Required    Description
id  yes identifies a document where to load a data from

API response includes the following fields:
Name    Description
table   tabular view of the financial data
pre_text    text that appears before the table in a document, describing what is a a table
post_text   a text that appears after the table in a document, providing analysis of the data in a table
"""

dotenv.load_dotenv()


def read_data(file="data/train.json"):
    documents = {}
    with open(file) as f:
        raw_data = f.read()
        data = json.loads(raw_data)

        for record in data:
            document_id = record.get("id").replace("/", "_")
            if not record.get("qa", {}).get("question"):
                continue
            if document_id not in documents:
                documents[document_id] = {
                    "id": document_id,
                    "filename": record.get("filename"),
                    "pre_text": record.get("pre_text"),
                    "post_text": record.get("post_text"),
                    "table": record.get("table"),
                    "table_ori": record.get("table_ori"),
                    "question": record.get("qa", {}).get("question"),
                    "answer": record.get("qa", {}).get("answer"),
                }
            else:
                print("WARNING id already included")

        return documents


def get_weather(location):
    chain = APIChain.from_llm_and_api_docs(
        OpenAI(temperature=0),
        open_meteo_docs.OPEN_METEO_DOCS,
        verbose=True,
        limit_to_domains=["https://api.open-meteo.com/"],
    )
    chain.invoke(f"What is the weather like right now in {location}?")


def get_api_chain(llm_strategy: LLMStrategy, domains=None, verbose=True):

    _domains = ["http://127.0.0.1:8001/"]
    _llm = llm_strategy.get_llm()

    if domains:
        _domains = domains

    return APIChain.from_llm_and_api_docs(
        llm=_llm,
        api_docs=LOCAL_API_DOCS,
        verbose=verbose,
        limit_to_domains=_domains,
    )


def get_llm_chain(llm_strategy: LLMStrategy):
    _llm = llm_strategy.get_llm()
    prompt_template = "You are helpful assistant solving {type_of_problem} problems."
    prompt = PromptTemplate(
        input_variables=["type_of_problem"], template=prompt_template
    )
    return LLMChain(llm=_llm, prompt=prompt)


def get_prompt(prompt_strategy: PromptingStrategy, **args):
    prompt = prompt_strategy.get_prompt()
    prompt_formatted = prompt.format(**args)
    return prompt_formatted


def get_answer(chain, prompt):
    print(f"prompt: {prompt}")
    return chain.invoke(prompt)


def get_data(documents, document_id):
    data_table = documents.get(document_id).get("table")
    data_pre_text = ";".join(documents.get(document_id).get("pre_text"))
    data_post_text = ";".join(documents.get(document_id).get("post_text"))
    return f"{data_pre_text}\n{data_table}\n{data_post_text}"


def get_table(documents, document_id):
    return documents.get(document_id).get("table")
