FROM python:3.11.8

WORKDIR /app

ENV OPENAI_API_KEY=""

COPY pyproject.toml poetry.lock ./

RUN pip install poetry==1.8
RUN poetry install --no-root --no-interaction --no-ansi

WORKDIR /app/src

COPY ./src/ /app/src/

ENTRYPOINT ["poetry", "run", "python", "main.py"]
