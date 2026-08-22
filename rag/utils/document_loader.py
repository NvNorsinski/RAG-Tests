from pathlib import Path

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openpyxl import load_workbook

import config


class DocumentLoaderService:

    def load_file(self, file_path) -> list[Document]:
        """Load Files

        Args:
            file_path (str): Path to file

        Raises:
            ValueError: error message

        Returns:
            list[Document]: List with the Paths to the documents
        """        
        extension = file_path.suffix.lower()

        if extension == ".pdf":
            loader = PyPDFLoader(str(file_path))

        elif extension == ".docx":
            loader = Docx2txtLoader(str(file_path))

        elif extension == ".xlsx":
            return self._load_xlsx_file(file_path)

        elif extension == ".xls":
            loader = UnstructuredExcelLoader(str(file_path), mode="elements")

        elif extension in {".txt", ".md"}:
            loader = TextLoader(str(file_path), encoding="utf-8")

        else:
            raise ValueError(
                f"Unsupported file type: "
                f"{extension}"
            )

        documents = loader.load()
        # Ensure the source path is available

        for document in documents:
            document.metadata["source"] = str(file_path)
            document.metadata["file_type"] = extension

        return documents

    def _load_xlsx_file(self, file_path: Path) -> list[Document]:
        """Load Excel files

        Args:
            file_path (Path): Path of the file

        Raises:
            ValueError: Could not parse Excel file with openpyxl
            ValueError: No Content in the file

        Returns:
            list[Document]: list of Documents
        """        
        try:
            workbook = load_workbook(file_path, data_only=True, read_only=True)
        except Exception as error:
            raise ValueError(
                f"Could not parse Excel file with openpyxl: {file_path}"
            ) from error

        sheets = []
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                row_text = "\t".join(
                    "" if cell is None else str(cell)
                    for cell in row
                )
                rows.append(row_text)

            if rows:
                sheets.append(f"Sheet: {sheet.title}\n" + "\n".join(rows))

        if not sheets:
            raise ValueError(f"No readable content found in Excel file: {file_path}")

        return [
            Document(
                page_content="\n\n".join(sheets),
                metadata={
                    "source": str(file_path),
                    "file_type": ".xlsx",
                },
            )
        ]


    def load_all_documents(self):
        """Load documemnts

        Raises:
            FileNotFoundError: Error if File is not available
        Returns:
            list[Unknown]: List of files
        """        
        all_documents = []

        for file_path in (config.DOCUMENTS_DIR.rglob("*")):
            if not file_path.is_file():
                continue

            extension = file_path.suffix.lower()
            if (extension not in config.SUPPORTED_EXTENSIONS):
                continue
            try:
                documents = self.load_file(file_path)

                all_documents.extend(documents)

                print("Loaded:", file_path.name)

            except Exception as error:
                print("Could not load:", file_path.name)
                print(error)

        if not all_documents:
            raise FileNotFoundError("No supported documents were found.")

        return all_documents


    def create_chunks(self) -> list[Document]:
        """Create the Chunks of the documents

        Returns:
            list[Documents]: list of the chunks
        """        
        documents = self.load_all_documents()

        splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=(config.CHUNK_SIZE),
                chunk_overlap=(config.CHUNK_OVERLAP)
            )
        )

        chunks = splitter.split_documents(documents)

        print(f"Created {len(chunks)} chunks.")

        return chunks