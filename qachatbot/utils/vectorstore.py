from qachatbot import PERSIST_DIR, MD_PERSIST_DIR
from langchain_chroma import Chroma


class VectorStoreManager:
    def __init__(self, embedding_function) -> None:
        self.embedding_function = embedding_function
        self._chromadb = None

    @property
    def chromadb(self):
        if self._chromadb is None:
            self._chromadb = Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=self.embedding_function,
            )
        return self._chromadb

