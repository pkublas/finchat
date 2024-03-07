from llms import OpenAIStrategy
from prompts import (
    PromptWithFewShots,
    PromptSimple, PromptAWithAPIDocs, PromptStepByStepInstructions, PromptExecutePseudoCode,
    PromptGeneratePseudoCode
)

from services import read_data, get_answer, get_api_chain, get_prompt, get_llm_chain

documents = read_data("data/train.json")

openai_strategy = OpenAIStrategy()

chain_api = get_api_chain(llm_strategy=openai_strategy)
chain_llm = get_llm_chain(llm_strategy=openai_strategy)

prompt_with_few_shots = PromptWithFewShots()
prompt_simple = PromptSimple()
prompt_with_api_docs = PromptAWithAPIDocs()
prompt_step_by_step_guide = PromptStepByStepInstructions()

prompt_generate_pseudo_code = PromptGeneratePseudoCode()
prompt_execute_pseudo_code = PromptExecutePseudoCode()

# document_id = "Single_JKHY_2009_page_28.pdf-3"
document_id = "Single_HIG_2004_page_122.pdf-2"

prompt1 = get_prompt(prompt_strategy=prompt_generate_pseudo_code, **{
    "document_id": document_id,
    "question": documents.get(document_id).get("question")
})
answer1 = get_answer(
    chain=chain_api,
    prompt=prompt1
)
print(f"answer 1: {answer1.get('output')}")


# prompt2 = get_prompt(prompt_strategy=prompt_execute_pseudo_code, **{
#     "document_id": document_id,
#     "question": documents.get(document_id).get("question"),
#     "pseudo_code": answer1.get("output")
# })
# answer2 = get_answer(
#     chain=chain_api,
#     prompt=prompt1
# )
# print(f"answer 2: {answer2.get('output')}")
