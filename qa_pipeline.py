# qa_pipeline.py

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def get_qa_chain():
    # Load vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory="./chroma_langchain_db"
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Load LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # Prompt template
    template = """Use the following pieces of context from my lprfile to answer the question at the end.
    If you don't know the answer based on the provided context, just say that you don't know.
    Do not make up an answer or use outside knowledge.
    Be concise and clear in your response.

    Context:
    {context}

    Question: {question}

    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    return qa_chain
