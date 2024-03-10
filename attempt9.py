from langchain.chains import LLMChain, LLMMathChain
from langchain_core.language_models import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Bedrock

from services import read_data

MODEL = "amazon.titan-text-express-v1"
DESCRIPTION = "Using GPT4All and the same set of prompts as with gpt4boto3"


class FinancialAnalysisChain(LLMChain):

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = True) -> LLMChain:
        analysis_prompt = """You work as financial analyst helping your company understand financial report.
        Answer the questions quoted in triple backticks below given available data between `===`.
        ```{question}```
        ===
        {data}
        ===
        """

        prompt = PromptTemplate(
            template=analysis_prompt, input_variables=["data", "question"]
        )

        return cls(prompt=prompt, llm=llm, verbose=verbose)


def run(input):

    llm = Bedrock(credentials_profile_name="default", model_id=MODEL)

    question_analysis_chain = FinancialAnalysisChain.from_llm(llm)
    question_analysis_answer = question_analysis_chain.invoke(
        input={"data": input.get("data"), "question": input.get("question")}
    )

    return question_analysis_answer.get("text")


if __name__ == "__main__":

    documents = read_data("data/train.json")

    # ids = ["Single_HIG_2004_page_122.pdf-2", "Single_JKHY_2009_page_28.pdf-3"]
    ids = ["Single_UPS_2009_page_33.pdf-2"]

    llm = Bedrock(credentials_profile_name="default", model_id=MODEL)
    for document_id in ids:
        doc = documents.get(document_id)
        question = doc.get("question")
        table = doc.get("table")
        post_text = ";".join(doc.get("post_text"))
        pre_text = ";".join(doc.get("pre_text"))

        data = f"{pre_text}\n{table}\n{post_text}"

        answer = run(input={"data": data, "question": question})

        print(answer)
        print(answer.get("text"))
