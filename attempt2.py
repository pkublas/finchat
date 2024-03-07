import json
from urllib.request import urlopen

from langchain.chains import StuffDocumentsChain, LLMChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI


from services import read_data, get_answer

documents = read_data("data/train.json")

document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
document_variable_name = "context"
prompt = PromptTemplate.from_template(
    "Provide numexpr expression that answers a user's question quoted in triple backticks."
    "Data: {context}"
    "Question: ```{question}```"
)

llm = OpenAI()
llm_chain = LLMChain(llm=llm, prompt=prompt)

documents_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
)

document_id = "Single_JKHY_2009_page_28.pdf-3"
# document_id = "Single_HIG_2004_page_122.pdf-2"

data = None
with urlopen(f"http://127.0.0.1:8001/table?id={document_id}") as f:
    data = json.load(f)

chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_prompt=document_prompt,
    document_variable_name=document_variable_name,
)

# doc = Document(page_content=f"{data.get('page_content')} \n {data.get('table')} \n {data.get('post_text')}", metadata={"page": "1"})

page1 = Document(
    page_content=f"{data.get('pre_text')}",
    metadata={"page": "1", "description": "tabular view of the financial data"},
)
page2 = Document(
    page_content=f"{data.get('table')}",
    metadata={
        "page": "2",
        "description": "a text that appears before the table in a document, describing what is a a table",
    },
)
page3 = Document(
    page_content=f"{data.get('post_text')}",
    metadata={
        "page": "3",
        "description": "a text that appears after the table in a document, providing analysis of the data in a table",
    },
)

answer = chain.invoke(
    input={
        "question": documents.get(document_id).get("question"),
        "input_documents": [page1, page2, page3],
    }
)
print(answer)
print(answer.get("output_text"))
