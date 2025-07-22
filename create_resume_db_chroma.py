import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings 

# Define loaders for different file types
def load_and_chunk_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return split_documents(documents)

def load_and_chunk_docx(docx_path):
    try:
        loader = UnstructuredWordDocumentLoader(docx_path)
        documents = loader.load()
        return split_documents(documents)
    except ValueError as e:
        print(f"Skipping {docx_path}: {e}")
        return []
    
# Common text splitter
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        add_start_index=True,
    )
    return splitter.split_documents(documents)

# Walk through ./resume and chunk all supported files
resume_dir = './resume/'
all_chunks = []

for root, dirs, files in os.walk(resume_dir):
    for filename in files:
        file_path = os.path.join(root, filename)
        rel_path = os.path.relpath(file_path, start=resume_dir)

        if filename.endswith(".pdf"):
            print(f"Processing PDF: {rel_path}")
            chunks = load_and_chunk_pdf(file_path)

        elif filename.endswith(".docx"):
            print(f"Processing DOCX: {rel_path}")
            chunks = load_and_chunk_docx(file_path)

        else:
            continue 

        for chunk in chunks:
            chunk.metadata["source"] = rel_path
        all_chunks.extend(chunks)

print(f"\nTotal chunks created: {len(all_chunks)}")

# Build and persist Chroma DB
persist_dir = "./tmp/chroma_resume_db"
os.makedirs(persist_dir, exist_ok=True)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
db = Chroma.from_documents(documents=all_chunks, embedding=embedding_model, persist_directory=persist_dir)
db.persist()

print(f"\n Chroma DB created at: {persist_dir}")
