import csv
import os
from dataclasses import dataclass, fields, asdict
from datetime import datetime
import re

from attempt9 import run, MODEL, DESCRIPTION
from services import read_data

RESULT_OK = "OK"
RESULT_WRONG = "WRONG"


@dataclass
class Document:
    id: str
    filename: str
    table_ori: str
    table: str
    question: str
    answer: str
    pre_text: str
    table: str
    post_text: str

    @property
    def data(self):
        return f"{self.pre_text}\n{self.table}\n{self.post_text}"

    @property
    def document_id(self):
        return self.id

    @property
    def expected_answer(self):
        return self.answer


@dataclass
class Result:
    document_id: str
    question: str
    expected_answer: str
    actual_answer: str = None
    result: str = "N/A"
    error: str = "N/A"
    processing_duration_in_sec: int = 0


def get_documents(data_file: str = "data/train.json", document_ids: list = None, number_of_docs: int = None):
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
            Document(**documents.get(doc_id)) for doc_id in documents if doc_id in document_ids
        ]
    else:
        return [Document(**documents.get(doc_id)) for doc_id in documents]


def parse_answer(answer):
    """
    Parses an answer provided by LLM into a decimal

    :param answer:
    :return:
    """
    pattern = r"Answer: (-?\d+\.\d+)"

    # TODO sometime answer expected is integer, handle that

    match = re.search(pattern, answer)
    if match:
        decimal_value = match.group(1)
        return decimal_value
    else:
        raise ValueError("unable to extract decimal")


def compare_answers(act, exp):
    """
    Compares two answers and return True if equal or False otherwise

    :param act:
    :param exp:
    :return:
    """
    act = act.replace("%", "")
    exp = exp.replace("%", "")

    act = float(act)
    exp = float(exp)

    if act < 1:
        act = act * 100
    elif exp < 1:
        exp = exp * 100

    precision = min(len(str(act).split(".")[1]), len(str(exp).split(".")[1]))
    value1 = round(act, precision)
    value2 = round(exp, precision)

    return value1 == value2


def generate_html_report(file_name, header, data):
    """
    Generate HTML report at given file_name with info about model and results

    :param file_name:
    :param header:
    :param data:
    :return:
    """
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
        <h1>report</h1>
        <h2>{header.get("model")}</h2>
        <h3>{header.get("description")}</h3>
        </hr>
        <table border="1">
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

    total_ok = 0
    total_wrong = 0

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

            if key == "result" and getattr(item, key) == RESULT_OK:
                total_ok += 1
            elif key == "result" and getattr(item, key) == RESULT_WRONG:
                total_wrong += 1

        html_content += "</tr>"

    total_rows = len(data)
    ratio_ok = total_ok / total_rows
    ratio_wrong = total_wrong / total_rows
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


def process_document(document: Document):
    """
    For a document specified in the input, process it

    :param document:
    :return:
    """

    result = Result(document_id=document.document_id,
                    question=document.question,
                    expected_answer=document.expected_answer
                    )

    try:
        start_time = datetime.now()
        actual_answer = run(input={"data": document.data, "question": document.question})
        processing_duration = int((datetime.now() - start_time).total_seconds())
    except ValueError as e:
        print(e)
        result.error = e
        result.processing_duration_in_sec = (datetime.now() - start_time).total_seconds()
        result.result = RESULT_WRONG
        return result
    print(actual_answer)
    try:
        actual_answer_parsed = parse_answer(actual_answer)
    except ValueError as e:
        print(e)
        result.error = e
        result.actual_answer = actual_answer
        result.processing_duration_in_sec = processing_duration
        result.result = RESULT_WRONG
        return result
    try:
        answers_are_same = compare_answers(actual_answer_parsed, document.expected_answer)
    except AttributeError as e:
        print(e)
        result.error = e
        result.actual_answer = actual_answer
        result.processing_duration_in_sec = processing_duration
        result.result = RESULT_WRONG
        return result

    result.error = "" if answers_are_same else "Answers are different",
    result.actual_answer = actual_answer
    result.processing_duration_in_sec = processing_duration
    result.result = RESULT_OK if answers_are_same else RESULT_WRONG

    return result


def generate_reports(do_html=True, do_csv=True):
    """
    Wrapper around generating html and csv reports

    :param do_html:
    :param do_csv:
    :return:
    """

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename_html = os.path.join("reports", f"report-{timestamp}.html")
    filename_csv = os.path.join("reports", f"report-{timestamp}.csv")
    if do_html:
        generate_html_report(
            filename_html, header={"model": MODEL, "description": DESCRIPTION}, data=results
        )
    if do_csv:
        generate_csv(filename_csv, results)


if __name__ == "__main__":

    document_ids = ["Single_HIG_2004_page_122.pdf-2", "Single_JKHY_2009_page_28.pdf-3"]
    number_of_docs = None

    documents = get_documents(document_ids=document_ids, number_of_docs=number_of_docs)

    results = []
    for document in documents:

        result: Result = process_document(document)
        results.append(result)

    generate_reports()
