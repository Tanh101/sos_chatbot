from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import CTransformers
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

app = FastAPI()

model_file = "models/vinallama-7b-chat_q5_0.gguf"
embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
llm = CTransformers(
    model=model_file, model_type="llama", max_new_tokens=512, temperature=0.6
)

vectorstore = None


class QueryRequest(BaseModel):
    query: str


def document_loader(filepath: str):
    loader = TextLoader(filepath)
    documents = loader.load()
    return documents


def text_splitter(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    return docs


@app.get("/")
async def read_root():
    return {"message": "Hello World"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global vectorstore
    file_path = f"data/{file.filename}"

    # with open(file_path, "wb") as f:
    #     f.write(await file.read())

    documents = document_loader(filepath=file_path)
    docs = text_splitter(documents=documents)

    vectorstore = Chroma.from_documents(docs, embedding_model)
    return {"message": "File uploaded and processed successfully"}


@app.post("/conversations")
async def query_rag_chain(request: QueryRequest):
    global vectorstore
    if vectorstore is None:
        return JSONResponse(
            status_code=400,
            content={"message": "No documents available. Please upload a file first."},
        )

    template = """
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    Context: {context}
    Question: {question}

    Only return the helpful answer below and nothing else.
    Helpful answer:
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    llm_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}, max_tokens_limit=1024),
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt},
    )

    result = llm_chain.invoke({"query": request.query})
    answer = result["result"]

    return {"answer": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
