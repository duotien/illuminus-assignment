import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qachatbot import PROJECT_DIR, DB_PERSIST_DIR, DOCUMENTS_DIR

if os.path.exists(DB_PERSIST_DIR):
    import shutil
    shutil.rmtree(DB_PERSIST_DIR)

os.makedirs(DB_PERSIST_DIR, exist_ok=True)

doc_paths = [os.path.join(DOCUMENTS_DIR, f) for f in os.listdir(DOCUMENTS_DIR) if ".txt" in f]
doc_generator = (TextLoader(doc_path, encoding="utf-8") for doc_path in doc_paths)
docs = [loader.load()[0] for loader in doc_generator]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5")

vectorstore = Chroma.from_documents(
    documents=all_splits,
    embedding=embeddings,
    persist_directory=DB_PERSIST_DIR,
)
# %%
