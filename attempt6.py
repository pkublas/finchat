from langchain.chains import LLMChain, LLMMathChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

MODEL = "gpt-4"
DESCRIPTION = "Translate question into numexpr (via LLMChain), update it to include required data from a dataset (via LLMChain), execute numexpr using LLMMathChain chain"

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

    prompt3 = PromptTemplate.from_template(
        "Execute Python's numexpr expression and produce an answer."
        "Expression: {question}"
    )

    llm = ChatOpenAI(model=MODEL)
    llm_chain1 = LLMChain(llm=llm, prompt=prompt1)
    llm_chain2 = LLMChain(llm=llm, prompt=prompt2)
    llm_chain3 = LLMMathChain.from_llm(llm=llm)

    answer1 = llm_chain1.invoke(input={"data": data, "question": question})
    # print(f"answer1_keys={answer1.keys()}")
    # print(f"answer1={answer1}")
    # print(f"answer1={answer1.get('text')}")

    answer2 = llm_chain2.invoke(input={"data": data, "question": question})
    # print(f"answer2_keys={answer2.keys()}")
    # print(f"answer2={answer2}")
    # print(f"answer2={answer2.get('text')}")

    answer3 = llm_chain3.invoke(input={"question": answer2.get('text')})
    # print(f"answer3_keys={answer3.keys()}")
    # print(f"answer3={answer3}")
    # print(f"answer3={answer3.get('answer')}")

    return answer3.get('answer')


if __name__ == "__main__":
    # document_id = "Single_JKHY_2009_page_28.pdf-3"
    # document_id = "Single_HIG_2004_page_122.pdf-2"
    pass

