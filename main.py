import os
import dotenv

from agents import (
    AgentFinancialAnalystSimple,
    AgentFinancialAnalystWithHistory,
    AgentFinancialAnalystDeveloper,
)
from llms import OpenAIStrategy, BedrockStrategy
from models import Result
from utils import generate_reports, process_document, get_documents

PROVIDERS = {"openai": OpenAIStrategy, "bedrock": BedrockStrategy}

AGENTS = {
    "simple": AgentFinancialAnalystSimple,
    "agent_keeping_history": AgentFinancialAnalystWithHistory,
    "agent_developer": AgentFinancialAnalystDeveloper,
}

SAMPLE_DOCUMENTS = ["Single_HIG_2004_page_122.pdf-2", "Single_JKHY_2009_page_28.pdf-3"]

dotenv.load_dotenv()


if __name__ == "__main__":

    config = {**os.environ}
    number_of_docs = config.get("NO_OF_DOCUMENTS_TO_PROCESS", 0)
    verbose = config.get("VERBOSE")

    documents_to_process = get_documents(
        document_ids=SAMPLE_DOCUMENTS, number_of_docs=int(number_of_docs)
    )

    model_provider = config.get("MODEL_PROVIDER", "openai")
    model_name = config.get("MODEL_NAME", "gtp-4")
    agent = config.get("AGENT", "simple")

    print(f"model_provider={model_provider} | model_name={model_name} | agent={agent}")

    llm = PROVIDERS.get(model_provider)(model_name=model_name)
    agent = AGENTS.get(agent)(llm=llm, verbose=verbose)

    results = []
    for document in documents_to_process:
        agent.restart()
        result: Result = process_document(agent, document)
        results.append(result)

    generate_reports(
        results=results,
        header={
            "model_name": llm.model_name,
            "agent_name": agent.__class__.__name__,
            "description": agent.description
        }
    )
