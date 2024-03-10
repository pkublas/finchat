from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

from services import read_data


def get_data():
    data_table = documents.get(document_id).get("table")
    data_pre_text = ";".join(documents.get(document_id).get("pre_text"))
    data_post_text = ";".join(documents.get(document_id).get("post_text"))
    return f"{data_pre_text}\n{data_table}\n{data_post_text}"

def get_table():
    return documents.get(document_id).get("table")


documents = read_data("data/train.json")

prompt1 = PromptTemplate.from_template(
    "You are Python programmer helping the business to understand available data."
    "Your response must be a valid JSON format."
    "The main question posed in triple backticks below can be answered using the data."
    "Your task is to breakdown the main question"
    "into a set of no more than 5 questions that must be answered before the main one can be resolved."
    "Data: {data}"
    "Question: {question}"
)

prompt2 = PromptTemplate.from_template(
    "You are financial analyst working on a set of questions posed by the customer."
    "Provide answers to all questions below based on available data and table."
    "Questions: {question}"
    "Data: {data}"
    "Table: {table}"
)

prompt3 = PromptTemplate.from_template(
    "You are Python interpreter."
    "Generate and execute Python's numexpr expression to answer a question "
    "quoted in triple backticks below."
    "Question: ```{question}```"
    "Data: {data}"
)

llm = OpenAI(model="gpt-3.5-turbo-instruct")
llm_chain1 = LLMChain(llm=llm, prompt=prompt1)
llm_chain2 = LLMChain(llm=llm, prompt=prompt2)
llm_chain3 = LLMChain(llm=llm, prompt=prompt3)

# document_id = "Single_JKHY_2009_page_28.pdf-3"
document_id = "Single_HIG_2004_page_122.pdf-2"

print(f"{document_id}\n{documents.get(document_id).get('question')}")

answer1 = llm_chain1.invoke(input={"data": get_data(), "question": documents.get(document_id).get('question')})
print(f"answer1_keys={answer1.keys()}")
print(f"answer1={answer1}")
print(f"answer1={answer1.get('text')}")


answer2 = llm_chain2.invoke(input={"data": get_data(), "table": get_table(), "question": answer1.get('text')})
print(f"answer2_keys={answer2.keys()}")
print(f"answer2={answer2}")
print(f"answer2={answer2.get('text')}")


answer3 = llm_chain3.invoke(input={"data": answer2.get('text'), "question": documents.get(document_id).get('question')})
print(f"answer3_keys={answer3.keys()}")
print(f"answer3={answer3}")
print(f"answer3={answer3.get('text')}")
