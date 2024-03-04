from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


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

documents = read_data()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, limit: int = 10):
    return templates.TemplateResponse(
        request=request, name="index.html", context={"documents": [d for d in documents][:limit]}
    )


@app.get("/document/{id}", response_class=HTMLResponse)
async def read_document(request: Request, id: str):
    return templates.TemplateResponse(
        request=request, name="document.html", context={"document": documents[id]}
    )
