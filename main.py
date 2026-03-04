from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader
import io
import os
import uuid
import traceback

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

from dotenv import load_dotenv
from google import genai
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers=["*"]
)
app.mount("/app", StaticFiles(directory="static", html=True), name="static")
# ----------------------------
# ENV + Gemini (NO hardcoded keys)
# ----------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY not set. /ask will fail until you set it.")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

def call_gemini(prompt: str) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set in environment variables.")

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ----------------------------
# RAG components (load once)
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent Chroma (data won't vanish after restart)
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="docs")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# ----------------------------
# Schemas
# ----------------------------
class AskRequest(BaseModel):
    doc_id: str
    question: str

# ----------------------------
# Endpoints
# ----------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))

        text = ""
        for page in pdf_reader.pages:
            text += (page.extract_text() or "")

        chunks = splitter.split_text(text)

        if not chunks:
            return {
                "filename": file.filename,
                "doc_id": file.filename.strip(),
                "total_chunks": 0,
                "sample_chunk": "",
                "message": "No extractable text found in PDF."
            }

        embeddings = embed_model.encode(chunks).tolist()

        doc_id = file.filename.strip()
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        metadatas = [{"doc_id": doc_id, "chunk_index": i} for i in range(len(chunks))]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )

        return {
            "filename": file.filename,
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "sample_chunk": chunks[0],
            "db_total_vectors": collection.count()
        }

    except Exception as e:
        print("ERROR in /upload:", repr(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
async def ask(req: AskRequest):
    try:
        question = req.question.strip()
        doc_id = req.doc_id.strip()

        if not question:
            raise HTTPException(status_code=400, detail="question is empty")
        if not doc_id:
            raise HTTPException(status_code=400, detail="doc_id is empty")

        q_emb = embed_model.encode(question).tolist()

        results = collection.query(
            query_embeddings=[q_emb],
            n_results=3,
            where={"doc_id": doc_id}
        )

        retrieved_chunks = results.get("documents", [[]])[0]

        if not retrieved_chunks:
            # helpful debug: show count
            return {
                "answer": "",
                "retrieved_chunks": [],
                "message": f"No chunks found for doc_id='{doc_id}'.",
                "db_total_vectors": collection.count()
            }

        context = "\n".join(retrieved_chunks)

        prompt = f"""Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

        answer = call_gemini(prompt)

        return {
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }

    except HTTPException:
        raise

    except Exception as e:
        print("ERROR in /ask:", repr(e))
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))