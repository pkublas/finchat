from langchain.chains import LLMChain, LLMMathChain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import GPT4All

from utils import read_data


def run(data, question):

    prompt1 = PromptTemplate.from_template(
        "You are Python programmer helping the business to understand available data."
        "Your task is to translate the main question in triple backticks below into Python's numexpr expression"
        "using descriptive variable names"
        "Question: {question}"
    )

    prompt2 = PromptTemplate.from_template(
        "Update provided Python's numexpr expression to include required datas values using data set below"
        "Expression: {question}"
        "Data: {data}"
    )

    llm = GPT4All(
        model="C:/Users/Pawel Kublas/.cache/gpt4all/mistral-7b-openorca.gguf2.Q4_0.gguf",
        max_tokens=500,
    )

    llm_chain1 = LLMChain(llm=llm, prompt=prompt1)
    llm_chain2 = LLMChain(llm=llm, prompt=prompt2)
    llm_chain3 = LLMMathChain.from_llm(llm=llm)

    answer1 = llm_chain1.invoke(input={"data": data, "question": question})
    print(f"answer1={answer1}")

    answer2 = llm_chain2.invoke(input={"data": data, "question": question})
    print(f"answer2={answer2}")

    answer3 = llm_chain3.invoke(input={"question": answer2.get("text")})
    print(f"answer3={answer3}")

    return answer3.get("answer")


if __name__ == "__main__":

    documents = read_data("../data/train.json")
    document_id = "Single_UPS_2009_page_33.pdf-2"
    doc = documents.get(document_id)
    question = doc.get("question")
    pre_text = doc.get("pre_text")
    post_text = doc.get("post_text")
    table = doc.get("table")
    data = f"{pre_text}\n{table}\n{post_text}"

    answer = run(data=data, question=question)
    print(answer)
