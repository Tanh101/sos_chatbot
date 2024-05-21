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
from langchain.chains import RetrievalQA

model_file = "models/vinallama-7b-chat_q5_0.gguf"

embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")

llm = CTransformers(
    model=model_file, model_type="llama", max_new_tokens=512, temperature=0.5
)


def document_loader(filepath: str):
    loader = TextLoader(filepath)
    documents = loader.load()
    return documents


def text_splitter(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)
    return docs


file_path = "data/test.txt"
documents = document_loader(filepath=file_path)
docs = text_splitter(documents=documents)


vectorstore = Chroma.from_documents(
    docs,
    embedding_model,
)

retriever = vectorstore.as_retriever()


template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])

# rag_chain = (
#     {
#         "context": lambda x: retriever.get_relevant_documents(x["question"]),
#         "question": RunnablePassthrough(),
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )

llm_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 2}, max_tokens_limit=1024),
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt},
)

# Query example
# query = "Số điện thoại cứu nạn ở khu vực Hải Châu là bao nhiêu?"
query = "Thủ đô của việt nam tên là gì?"

result = llm_chain.invoke({"query": query})

answer = result["result"]

print(answer.split("\n")[0])

