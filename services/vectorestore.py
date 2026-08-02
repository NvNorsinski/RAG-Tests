import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

import config
from services.emeddings import EmbeddingService
from utils.document_loader import DocumentLoaderService


class VectorStoreService:
    """Stores the document in Vector database."""

    def __init__(self, recreate=False):
        self.embedding = EmbeddingService().get()
        self.persist_dir = Path(config.CHROMA_DIR)

        if recreate and self.persist_dir.exists():
            shutil.rmtree(self.persist_dir, ignore_errors=True)

        if self.persist_dir.exists() and any(self.persist_dir.iterdir()):
            try:
                self.db = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embedding
                )
            except Exception:
                shutil.rmtree(self.persist_dir, ignore_errors=True)
                self.db = self._create_database()
        else:
            self.db = self._create_database()

    def _create_database(self):
        """Creates and returns a Chroma database populated with text chunks.

        This method loads documents, splits them into chunks, and creates a Chroma
        vector database using the specified embedding model. The database is persisted
        to the directory specified by `self.persist_dir`.

        Returns:
            chromadb.api.client.Client: The Chroma database client instance.
        """   
        loader = DocumentLoaderService()
        chunks = loader.create_chunks()

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        return Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding,
            persist_directory=str(self.persist_dir)
        )

    def create_database(self):
        """creates the database for Documents

        Returns:
            chromadb.api.client.Client: The Database
        """
        self.vectorstore = self._create_database()
        return self.vectorstore

    def retriever(self) -> VectorStoreRetriever:
        """Defines the retriever

        Returns:
            VectorStoreRetriever: Configured MMR retriever instance.
        """
        return self.db.as_retriever(
            search_type="mmr",  # searches for more diverse documents, not just the most similar ones
            search_kwargs={
                "k": 4,  # amount of documents to return
                "fetch_k": 20,  # number of documents fetched to the search process
                "lambda_mult": 0.5  # diversity
            }
        )

   
  