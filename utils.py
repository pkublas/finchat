import csv
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
import re
from langchain.chains import APIChain

from llms import LLMStrategy
from models import Document, Result

RESULT_OK = "OK"
RESULT_WRONG = "WRONG"


def get_documents(
    data_file: str = "data/train.json",
    document_ids: list = None,
    number_of_docs: int = None,
):
    """
    Returns a list of specified document, as per document_ids
        or a set number of documents, per number_of_docs

    :param data_file:
    :param document_ids:
    :param number_of_docs:
    :return:
    """
    documents = read_data(data_file)

    if number_of_docs:
        return [
            Document(**documents.get(doc_id))
            for i, doc_id in enumerate(documents)
            if i < number_of_docs and documents.get(doc_id).get("question")
        ]
    elif document_ids:
        return [
            Document(**documents.get(doc_id))
            for doc_id in documents
            if doc_id in document_ids
        ]
    else:
        return [Document(**documents.get(doc_id)) for doc_id in documents]


def parse_answer(answer):
    """
    Parses an answer provided by LLM into a decimal

    :param answer:
    :return:
    """
    pattern1 = re.compile(r"Answer:\s*(-?\d+(\.\d+)?|\d+)")  # e.g. "Answer: 15.9%"
    pattern2 = re.compile(
        r"(-?\d+(\.\d+)?|\d+)%"
    )  # "e.g. from operating activities was 14.13%."

    match1 = re.search(pattern1, answer)
    match2 = re.search(pattern2, answer)
    if match1:
        return match1.group(1)
    elif match2:
        return match2.group(1)
    else:
        raise ValueError("unable to extract decimal")


def compare_answers(act, exp):
    """
    Compares two answers and return True if equal or False otherwise

    :param act:
    :param exp:
    :return:
    """
    actual = act.replace("%", "")
    expected = exp.replace("%", "")

    actual = float(actual)
    expected = float(expected)

    if "-" not in act and actual < 1:
        actual = actual * 100
    elif "-" not in exp and expected < 1:
        expected = expected * 100

    if "." not in exp:
        precision = 0
    else:
        precision = min(
            len(str(actual).split(".")[1]), len(str(expected).split(".")[1])
        )
    value1 = round(actual, precision)
    value2 = round(expected, precision)

    return value1 == value2


def generate_html_report(file_name, header, data):
    """
    Generate HTML report at given file_name with info about model and results

    :param file_name:
    :param header:
    :param data:
    :return:
    """

    total_ok = 0
    total_wrong = 0
    for item in data:
        if getattr(item, "result") == RESULT_OK:
            total_ok += 1
        elif getattr(item, "result") == RESULT_WRONG:
            total_wrong += 1

    total_rows = len(data)
    ratio_ok = total_ok / total_rows
    ratio_wrong = total_wrong / total_rows

    html_content = """
    <html>
    <head>
        <title>HTML Report</title>
        <style>
            table { border-collapse: collapse; width: 100%; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            tr:hover { background-color: #ddd; }
        </style>
    </head>
    <body>
    """
    html_content += f"""
        <h1>Report</h1>
        
        <table border="0">
            <tr><th>Model</th><th>Agent</th><th>Description</th></tr>
            <tr>
                <td>{header.get("model_name")}</td>
                <td>{header.get("agent_name")}</td>
                <td>{header.get("description")}</td></tr>
        </table>
        
        <h3></h3>

        <table border="0">
            <tr><th>No. of documents processed</th><th>Success rate</th><th>Error rate</th></tr>
            <tr>
                <td>{total_rows}</td>
                <td>{ratio_ok * 100:.2f}</td>
                <td>{ratio_wrong * 100:.2f}</td></tr>
        </table>
        
        <h3></h3>
        
        <table border="0">
            <tr>
                <th>Document Id</th>
                <th>Question</th>
                <th>Answer Expected</th>
                <th>Actual Answer</th>
                <th>Result</th>
                <th>Processing Duration [sec]</th>
                <th>Error</th>
            </tr>
    """

    for item in data:
        html_content += "<tr>"
        for key in [
            "document_id",
            "question",
            "expected_answer",
            "actual_answer",
            "result",
            "processing_duration_in_sec",
            "error",
        ]:
            html_content += f"<td>{getattr(item, key)}</td>"

        html_content += "</tr>"

    html_content += (
        f"<tr><td colspan='7'>Number of documents: {total_rows} | "
        f"Success rate: {ratio_ok * 100:.2f} | "
        f"Error rate: {ratio_wrong * 100:.2f}</td></tr>"
    )
    html_content += "</table></body></html>"

    html_content += """
        </table>
    </body>
    </html>
    """

    with open(file_name, "w") as html_file:
        html_file.write(html_content)

    print(f"HTML saved to {file_name}")


def generate_csv(file_name, data):
    """
    Save results to CSV, for potential further analysis
    :param file_name:
    :param data:
    :return:
    """

    with open(file_name, "w", newline="") as csvfile:
        fieldnames = asdict(data[0]).keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(asdict(row))

    print(f"CSV saved to {file_name}")


def process_document(agent, document: Document):
    """
    For a document specified in the input, process it

    :param document:
    :return:
    """

    print(f"* document_id={document.document_id}")
    result = Result(
        document_id=document.document_id,
        question=document.question,
        expected_answer=document.expected_answer,
    )

    try:
        start_time = datetime.now()
        actual_answer = agent.start(
            input={"data": document.data, "question": document.question}
        )
        processing_duration = int((datetime.now() - start_time).total_seconds())
    except ValueError as e:
        print(e)
        result.error = e
        result.processing_duration_in_sec = (
            datetime.now() - start_time
        ).total_seconds()
        result.result = RESULT_WRONG
        return result
    try:
        actual_answer_parsed = parse_answer(actual_answer)
    except (ValueError, TypeError) as e:
        print(e)
        result.error = e
        result.actual_answer = actual_answer
        result.processing_duration_in_sec = processing_duration
        result.result = RESULT_WRONG
        return result
    try:
        answers_are_same = compare_answers(
            actual_answer_parsed, document.expected_answer
        )
    except AttributeError as e:
        print(e)
        result.error = e
        result.actual_answer = actual_answer
        result.processing_duration_in_sec = processing_duration
        result.result = RESULT_WRONG
        return result

    result.error = "" if answers_are_same else "Answers are different"
    result.actual_answer = actual_answer
    result.processing_duration_in_sec = processing_duration
    result.result = RESULT_OK if answers_are_same else RESULT_WRONG

    return result


def generate_reports(results, header, do_html=True, do_csv=True):
    """
    Wrapper around generating html and csv reports

    :param do_html:
    :param do_csv:
    :return:
    """

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_html = os.path.join("reports", f"report-{header.get('model_name')}-{timestamp}.html")
    filename_csv = os.path.join("reports", f"report-{header.get('model_name')}-{timestamp}.csv")
    if do_html:
        generate_html_report(
            filename_html,
            header=header,
            data=results,
        )
    if do_csv:
        generate_csv(filename_csv, results)


def read_data(file="data/train.json"):
    documents = {}
    with open(file) as f:
        raw_data = f.read()
        data = json.loads(raw_data)

        for record in data:
            document_id = record.get("id").replace("/", "_")
            if not record.get("qa", {}).get("question"):
                continue
            if document_id not in documents:
                documents[document_id] = {
                    "id": document_id,
                    "filename": record.get("filename"),
                    "pre_text": record.get("pre_text"),
                    "post_text": record.get("post_text"),
                    "table": record.get("table"),
                    "table_ori": record.get("table_ori"),
                    "question": record.get("qa", {}).get("question"),
                    "answer": record.get("qa", {}).get("answer"),
                }
            else:
                print("WARNING id already included")

        return documents


LOCAL_API_DOCS = """
BASE URL: http://127.0.0.1:8001

API Documentation

API provides financial data from variety of documents at /table.

Query string parameters:
Name    Required    Description
id  yes identifies a document where to load a data from

API response includes the following fields:
Name    Description
table   tabular view of the financial data
pre_text    text that appears before the table in a document, describing what is a a table
post_text   a text that appears after the table in a document, providing analysis of the data in a table
"""


def get_api_chain(llm_strategy: LLMStrategy, domains=None, verbose=True):
    """chain working with API retriever"""

    _domains = ["http://127.0.0.1:8001/"]
    _llm = llm_strategy.get_llm()

    if domains:
        _domains = domains

    return APIChain.from_llm_and_api_docs(
        llm=_llm,
        api_docs=LOCAL_API_DOCS,
        verbose=verbose,
        limit_to_domains=_domains,
    )
