import json
from urllib.request import urlopen

from langchain.chains import LLMMathChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from services import read_data

documents = read_data("data/train.json")

prompt = PromptTemplate.from_template(
    "Answer user's question based on JSON formatted data from a financial document below."
    "Provided JSON data includes the following fields."
    "table - tabular view of the financial data"
    "pre_text - text that appears before the table in a document, describing what is a a table"
    "post_text   a text that appears after the table in a document, providing analysis of the data in a table"
    "Data: {context}"
    "Question: ```{question}```"
)

llm = OpenAI()
llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)

# document_id = "Single_JKHY_2009_page_28.pdf-3"
document_id = "Single_HIG_2004_page_122.pdf-2"

with urlopen(f"http://127.0.0.1:8001/table?id={document_id}") as f:
    data = json.load(f)

answer = llm_math.invoke(
    input={
        "question": documents.get(document_id).get("question"),
        "context": data
    }
)

# answer = llm_math.invoke(
#     input={
#         "question": "what is the ratio of revenue between years 2008 and 2009",
#         "context": "revenue in year 2008 was 100, and in 2009 78"
#     }
# )

print(json.dumps(answer, indent=4))
print(answer.get('output_text'))
