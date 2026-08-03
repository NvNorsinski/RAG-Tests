import shutil

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

import config
from services.embeddings import EmbeddingService
from utils.document_loader import DocumentLoaderService


class VectorStoreService:

    def __init__(self):

        self.embeddings = EmbeddingService().get_embeddings()
        self.vectorstore = None
        self.chunks = None


  
    def get_chunks(self):
        """Returns Document chunks

        Returns:
           list[Document] : List of Chunks of the Documents
        """        
        if self.chunks is None:
            loader = DocumentLoaderService()
            self.chunks = loader.create_chunks()
            
        return self.chunks



    def database_exists(self) -> bool:
        """return database path if Database already exists

        Returns:
            bool: if Path of Chroma DB databases exists
        """        
        return (
            config.CHROMA_DIR.exists()
            and any(
                config.CHROMA_DIR.iterdir()
            )
        )


   

    def load_database(self) -> Chroma:
        """Load the Databass

        Returns:
            Chroma database: Chroma Database
        """        
        self.vectorstore = Chroma(persist_directory=str(config.CHROMA_DIR),
            embedding_function=self.embeddings
        )

        return self.vectorstore




    def create_database(self) -> Chroma:
        """Create databse if not exists

        Returns:
            Chroma database: Chroma Database
        """        

        chunks = self.get_chunks()

        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        self.vectorstore = (
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(
                    config.CHROMA_DIR
                )
            )
        )

        return self.vectorstore



    def get_vectorstore(self) -> Chroma:
        """ Returns:
        Chroma: The vector store instance, either existing, loaded, or newly 

        Returns:
            Chroma: The vector store instance, either existing, loaded, or newly 
        """        
        if self.vectorstore is not None:
            return self.vectorstore

        if self.database_exists():
            return self.load_database()
        return self.create_database()




    def get_retriever(self) -> EnsembleRetriever:

        chunks = self.get_chunks()

        # Vector Search + MMR

        vector_retriever = (
            self.get_vectorstore().as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": config.RETRIEVER_K,
                    "fetch_k": (
                        config.RETRIEVER_FETCH_K
                    ),
                    "lambda_mult": (
                        config.MMR_LAMBDA
                    )
                }
            )
        )

        # BM25

        bm25_retriever = BM25Retriever.from_documents(chunks)
        

        bm25_retriever.k = config.RETRIEVER_K

        # Hybrid Search

        hybrid_retriever = (
            EnsembleRetriever(
                retrievers=[
                    vector_retriever,
                    bm25_retriever
                ],
                weights=[
                    config.VECTOR_WEIGHT,
                    config.BM25_WEIGHT
                ]
            )
        )

        return hybrid_retriever




    def rebuild_database(self) -> Chroma:

        if config.CHROMA_DIR.exists():
            shutil.rmtree(config.CHROMA_DIR)

        self.vectorstore = None

        self.chunks = None

        return self.create_database()

