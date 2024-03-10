from dataclasses import dataclass


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
        pre_text = "\n".join(self.pre_text)
        post_text = "\n".join(self.post_text)
        return f"{pre_text}\n{self.table}\n{post_text}"

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
