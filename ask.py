from prompts import (
    PromptWithFewShots,
    PromptSimple, PromptAWithAPIDocs, PromptStepBySpteInstructions
)

from services import read_data, get_answer, get_local_chain

documents = read_data("data/train.json")
chain_local_api = get_local_chain()

prompt_with_few_shots = PromptWithFewShots()
prompt_simple = PromptSimple()
prompt_with_api_docs = PromptAWithAPIDocs()
prompt_step_by_step_guide = PromptStepBySpteInstructions()

# document_id = "Single_JKHY_2009_page_28.pdf-3"
document_id = "Single_HIG_2004_page_122.pdf-2"

answer = get_answer(
    prompt_strategy=prompt_step_by_step_guide,
    chain=chain_local_api,
    document_id=document_id,
    question=documents.get(document_id).get("question")
)
print(answer.get("output"))
