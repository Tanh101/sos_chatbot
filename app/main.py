# langchain
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
import uuid
from langchain_community.llms import CTransformers
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from langchain.chains import RetrievalQA
import os

# gemini import
from langchain_google_genai import ChatGoogleGenerativeAI
from constants import constants
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embedding_model = GPT4AllEmbeddings(model_file=constants.embedding_model_file)

llm = ChatGoogleGenerativeAI(
    model=constants.gg_model_name, google_api_key=constants.gg_api_key, temperature=0.1
)

vectorstore = None


class QueryRequest(BaseModel):
    query: str


class AnswerResponse(BaseModel):
    answer: str


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
    return {"message": "Server is running"}


file_path = "/home/vantanhly/CodingLife/langchain/sos_chatbot/app/data/test.txt"
documents = document_loader(filepath=file_path)
docs = text_splitter(documents=documents)


vectorstore = Chroma.from_documents(
    docs,
    embedding=embedding_model,
)


@app.post("/api/conversations")
async def query_rag_chain(request: QueryRequest):
    global vectorstore
    if vectorstore is None:
        return JSONResponse(
            status_code=400,
            content={"message": "No documents available. Please upload a file first."},
        )

    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    Use three sentences maximum and keep the answer concise. If the question not found in the context, please provide your own answer.
    Question: {question}
    Context: {context}
    Answer:
    """

    llm_template = """You are an assistant for question-answering tasks. Please provide the asnwer using vietnamese
    Question: {question}
    """

    llm_prompt = ChatPromptTemplate.from_template(llm_template)

    retriever = vectorstore.as_retriever()
    # prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # llm_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type="stuff",
    #     retriever=vectorstore.as_retriever(
    #         search_kwargs={"k": 2}, max_tokens_limit=1024
    #     ),
    #     return_source_documents=False,
    #     chain_type_kwargs={"prompt": prompt},
    # )

    prompt = ChatPromptTemplate.from_template(template)

    lmm_chain = llm_prompt | llm | StrOutputParser()

    # rag_chain = (
    #     {
    #         "context": lambda x: retriever.get_relevant_documents(x["question"]),
    #         "question": RunnablePassthrough(),
    #     }
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # result = rag_chain.invoke({"question": request.query})
    # if result == "":
    result = lmm_chain.invoke({"question": request.query})
    answer = AnswerResponse(answer=result)

    return  {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
