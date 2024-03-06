from langchain.chains import APIChain
from langchain.chains.api import open_meteo_docs
from langchain_openai import OpenAI
import dotenv
import json

from prompts import PromptingStrategy

LOCAL_API_DOCS = """
BASE URL: http://127.0.0.1:8001

API Documentation

API provides financial data from variety of documents at /table.

Query string parameters:
Name    Required    Description
id  yes identifies a document where to load a data from

"""

dotenv.load_dotenv()


def read_data(file="data/train.json"):
    documents = {}
    with open(file) as f:
        raw_data = f.read()
        data = json.loads(raw_data)

        for record in data:
            document_id = record.get("id").replace("/", "_")
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


def get_local_chain(llm_strategy=None, domains=None, verbose=True):

    _domains = ["http://127.0.0.1:8001/"]
    _llm = OpenAI(temperature=0)

    if domains:
        _domains = domains

    if llm_strategy:
        _llm = llm_strategy

    return APIChain.from_llm_and_api_docs(
        llm=_llm,
        api_docs=LOCAL_API_DOCS,
        verbose=verbose,
        limit_to_domains=_domains,
    )


def get_answer(prompt_strategy: PromptingStrategy, chain, document_id, question):
    prompt = prompt_strategy.get_prompt()
    print(prompt.format(document_id=document_id, question=question))
    return chain.invoke(prompt.format(document_id=document_id, question=question))
