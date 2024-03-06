from fastapi import FastAPI

from services import read_data

app = FastAPI()

documents = read_data("data/train.json")



@app.get("/document/")
async def read_document(id: str):
    document = documents.get(id)
    return document


@app.get("/table")
async def read_table(id: str, name: str | None = "table"):
    document = documents.get(id)
    table = document.get(name)
    pre_text = document.get("pre_text")
    post_text = document.get("post_text")
    return {
        "table": table,
        "pre_text": pre_text,
        "post_text": post_text
    }
