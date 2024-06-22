# Necessary imports
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from constants import constants
import os
from langchain.chains import RetrievalQA

app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = GPT4AllEmbeddings(model_file=constants.embedding_model_file)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=constants.gg_api_key, temperature=0.1
)

vectorstore = None


class QueryRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str


def document_loader(filepath: str):
    if filepath.endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    else:
        loader = TextLoader(filepath)
    documents = loader.load()
    return documents


def text_splitter(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    return docs


def pdf_splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    return docs


@app.get("/")
async def read_root():
    return {"message": "Server is running"}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["text/plain", "application/pdf"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only plain text and PDF files are allowed.",
        )

    upload_dir = "./data"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, file.filename)

    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())

    documents = document_loader(filepath=file_location)

    if file.content_type == "application/pdf":
        docs = pdf_splitter(documents=documents)
    else:
        docs = text_splitter(documents=documents)

    db = Chroma.from_documents(
        docs, embedding=embedding_model, persist_directory="./chroma_db"
    )

    return {"detail": "File uploaded and processed successfully"}


@app.post("/api/conversations")
async def query_rag_chain(request: QueryRequest):
    template = """You are an assistant for rescue information tasks.
    Answer in Vietnamese. This is important.
    if you don't know the answer, please say "I don't know".
    Question: {question}
    Context: {context}
    Answer:
    """

    llm_template = """You are an assistant for something information.
    Answer in Vietnamese. This is important.
    Question: {question}
    Answer:
    """

    # only llm
    llm_prompt = ChatPromptTemplate.from_template(llm_template)
    lmm_chain = llm_prompt | llm | StrOutputParser()

    # retrieval
    vectorstore = Chroma(
        persist_directory="./chroma_db", embedding_function=embedding_model
    )
    retriever = vectorstore.as_retriever()
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        verbose=True,
        return_source_documents=True,
    )

    result = rag_chain.invoke({"query": request.query})

    if "không biết" in result:
        result = lmm_chain.invoke({"question": request.query})

    # Format the answer to replace '\n' with actual newlines
    
    answer = AnswerResponse(answer=result)

    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
