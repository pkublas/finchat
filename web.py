from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from services import read_data

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")

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
