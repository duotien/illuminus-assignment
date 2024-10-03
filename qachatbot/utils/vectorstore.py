from langchain_chroma import Chroma

from qachatbot import DB_PERSIST_DIR


class VectorStoreManager:
    def __init__(self, embedding_function) -> None:
        self.embedding_function = embedding_function
        self._chromadb = None

    @property
    def chromadb(self):
        if self._chromadb is None:
            self._chromadb = Chroma(
                persist_directory=DB_PERSIST_DIR,
                embedding_function=self.embedding_function,
            )
        return self._chromadb
