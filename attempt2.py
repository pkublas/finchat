import json
import sys
from urllib.request import urlopen

from langchain.chains import StuffDocumentsChain, LLMChain, LLMMathChain
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI


from services import read_data, get_answer

documents = read_data("data/train.json")

document_prompt = PromptTemplate(
    input_variables=["page_content"], template="{page_content}"
)
document_variable_name = "context"

prompt1 = PromptTemplate.from_template(
    "You are financial advisor."
    "Given the data formatted in JSON below"
    "Produce a detailed and descriptive report covering all values from `table` key"
    "along with a text that is preceding that table in 'pre_text'"
    "as well as a text that is following that table in 'post_text'"
    ""
    "Data: ```{context}```"
)

# prompt2 = PromptTemplate.from_template(
#     "Breakdown user's question quoted in triple backticks below into "
#     "several smaller questions that need to be answered in order to obtain the final answer."
#     "Question: ```{context}```"
# )

# prompt2 = PromptTemplate.from_template(
#     "You are financial advisor."
#     "Re-write user's question quoted in triple backticks to include required data points from the report below."
#     "Report: {data}"
#     "Question: ```{context}```"
# )

prompt3 = PromptTemplate.from_template(
    "You are Python developer."
    "Translate a question quoted in triple backticks below into a single expression using Python's numexpr library and available data."
    "{question}"
)

prompt4 = PromptTemplate.from_template(
    "You are Python interpreter. "
    "Execute Python's numexpr expression provided in triple backticks below and return an answer."
    "{question}"
)

llm = OpenAI()
llm_chain1 = LLMChain(llm=llm, prompt=prompt1)
# llm_chain2 = LLMChain(llm=llm, prompt=prompt2)
llm_chain3 = LLMChain(llm=llm, prompt=prompt3)
# llm_chain3 = LLMMathChain.from_llm(llm=llm, prompt=prompt3)
llm_chain4 = LLMMathChain.from_llm(llm=llm)

document_id = "Single_JKHY_2009_page_28.pdf-3"
# document_id = "Single_HIG_2004_page_122.pdf-2"

data = None
with urlopen(f"http://127.0.0.1:8001/table?id={document_id}") as f:
    data = json.load(f)

if not data:
    print("ERROR, unable to retrieve data from API")
    sys.exit(1)

# chain = StuffDocumentsChain(
#     llm_chain=llm_chain1,
#     document_prompt=document_prompt,
#     document_variable_name=document_variable_name,
# )

# page1 = Document(
#     page_content=f"{data.get('pre_text')}",
#     metadata={"page": "1", "description": "tabular view of the financial data"},
# )
# page2 = Document(
#     page_content=f"{data.get('table')}",
#     metadata={
#         "page": "2",
#         "description": "a text that appears before the table in a document, describing what is a a table",
#     },
# )
# page3 = Document(
#     page_content=f"{data.get('post_text')}",
#     metadata={
#         "page": "3",
#         "description": "a text that appears after the table in a document, providing analysis of the data in a table",
#     },
# )

# answer1 = chain.invoke(
#     input={
#         "input_documents": [page1, page2, page3],
#     }
# )

# answer1 = llm_chain1.invoke(input={"context": data})
#
# print(f"answer1_keys={answer1.keys()}")
# print(f"answer1={answer1}")
# print(f"answer1={answer1.get('text')}")

# answer2 = llm_chain2.invoke(
#     input={"context": documents.get(document_id).get("question"), "data": answer1.get('text')}
# )
#
# print(f"answer2_keys={answer2.keys()}")
# print(f"answer2={answer2}")
# print(f"answer2={answer2.get('text')}")

answer3 = llm_chain3.invoke(
    input={
        "question": f"Data: {data}, Question: ```{documents.get(document_id).get('question')}```"
    }, verbose=True
)

print(f"answer3_keys={answer3.keys()}")
print(answer3.get("text"))

answer4 = llm_chain4.invoke(
    input={
        "question": f"```{answer3.get('text')}```"
    }, verbose=True
)

print(f"answer4_keys={answer4.keys()}")
print(answer4.get("answer"))
