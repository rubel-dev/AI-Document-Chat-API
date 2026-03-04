FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pypdf \
    langchain-text-splitters \
    sentence-transformers \
    chromadb \
    python-dotenv \
    google-genai

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]