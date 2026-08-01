import os

from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

import config
from services.emeddings import EmbeddingService
from utils.pdf_loader import PdfLoader


class VectorStoreService:
    """Stores the document in Vector database."""    

    def __init__(self):
        self.embedding = EmbeddingService().get()

        if os.path.exists(config.CHROMA_DIR):
            self.db = Chroma(
                persist_directory=config.CHROMA_DIR,
                embedding_function=self.embedding
            )
        else:
            docs = PdfLoader().load()
            self.db = Chroma.from_documents(
                docs,
                self.embedding,
                persist_directory=config.CHROMA_DIR
            )

    def retriever(self) -> VectorStoreRetriever:
        """
        Defines retriever.
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