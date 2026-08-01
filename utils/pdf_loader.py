from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


class PdfLoader:

    def load(self):
        """Loads and splitts the PDF file into chunks

        Returns:
        List[Document]: Splitted Document

        """        

        loader = PyPDFLoader(config.PDF_FILE)

        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

        return splitter.split_documents(docs)