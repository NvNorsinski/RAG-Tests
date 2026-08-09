
from rag.services.embeddings import EmbeddingService
from rag.utils.document_loader import DocumentLoaderService


class VectorStoreService:

    def __init__(self):
        self.embeddings = EmbeddingService().get()

        self.vectorstore = None

        self.chunks = None


    def get_chunks(self):
        if self.chunks is None:
            loader = DocumentLoaderService()
            self.chunks = loader.create_chunks()
        return self.chunks