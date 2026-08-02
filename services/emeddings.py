from langchain_ollama import OllamaEmbeddings

import config


class EmbeddingService:
    """Loads the model
    
    """    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=config.EMBEDDING_MODEL
        )

    def get(self):
        """Get the embeddings for the model

        Returns:
            embddings: Ollama embeddings
        """        
        return self.embeddings