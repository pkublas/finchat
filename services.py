import json


def read_data(file="data/train.json"):
    documents = {}
    with open(file) as f:
        raw_data = f.read()
        data = json.loads(raw_data)

        for record in data:
            document_id = record.get("id").replace("/", "_")
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
