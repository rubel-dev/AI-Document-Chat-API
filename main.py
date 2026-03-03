from fastapi import FastAPI, UploadFile, File
from langchain_text_splitters import RecursiveCharacterTextSplitter

 

from pypdf import PdfReader
import io
app = FastAPI()

@app.post('/upload')

async def upload_pdf(file: UploadFile = File(...)):
   
        contents = await file.read()
        pdf_reader = PdfReader(io.BytesIO(contents))

        text = ""
        for page in pdf_reader.pages:
                text += page.extract_text()
        splitter = RecursiveCharacterTextSplitter(
                chunk_size = 500,
                chunk_overlap=100
        )
        chunks = splitter.split_text(text)
        print(len(chunks))
        print(chunks[0])
        
        return {
        "total_chunks": len(chunks),
        "first_chunk": chunks[0]
        }
                

         

    