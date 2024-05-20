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

model_file = "models/vinallama-7b-chat_q5_0.gguf"

embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")

llm = CTransformers(
    model=model_file, model_type="llama", max_new_tokens=100, temperature=0.01
)

def document_loader(filepath: str):
    loader = TextLoader(filepath)
    documents = loader.load()
    return documents


def text_splitter(documents):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
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
template = """<|im_start|>system\nYou are an AI assistant. Please provide stored information about the rescue information provided
to answer user questions. Please answer in the same language as the question. For example, a user asks What is the phone number of
nearest rescuer? Then the answer will be in English and it will be: The phone number of nearest rescuer is 0123456
.On the contrary, if the question is in Vietnamese, your answer must also be in Vietnamese.\n
{context}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant"""


prompt = PromptTemplate(template = template, input_variables=["context", "question"])

rag_chain = (
    {
        "context": lambda x: retriever.get_relevant_documents(x["question"]),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# Query example
query = "Trường hợp muốn cứu nạn ở khu vực Hải Châu thì có thể gọi số điện thoại nào?"
result = rag_chain.invoke({"question": query})

print(result)
