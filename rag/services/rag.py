from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import config
from rag.services.vectorestore import VectorStoreService


class RagService:
    def __init__(self):

        self.history = []

        self.retriever = VectorStoreService().get_retriever()

        self.llm = ChatOllama(
            model=config.LLM_MODEL,
            temperature=0,
            streaming=True
        )

        self.prompt = ChatPromptTemplate.from_template(
        """
        Du bist ein Dokumenten-Assistent.

        Chatverlauf:
        {history}

        Kontext:
        {context}

        Frage:
        {question}

        Antworte ausschließlich anhand des Kontextes. 
        Wenn Informationen nicht vorhanden sind gib dies an.
        """
        )

    def format_docs(self, docs):
        """Formats the Document for output.

        Args:
            docs (list[Document]): List of returns 
        Returns:
            str: formated String
        """        
            
        formatted_docs = []

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get(
                "source",
                "Unbekannte Datei"
            )

            file_type = doc.metadata.get(
                "file_type",
                "")

            page = doc.metadata.get("page")

            sheet_name = doc.metadata.get("page_name")

            location = ""

            # PDF: Seitennummer anzeigen
            if page is not None:
                location = f"Seite {page + 1}"

            # Excel: Tabellenblatt anzeigen
            elif sheet_name:
                location = (
                    f"Tabellenblatt "
                    f"{sheet_name}"
                )

            # Word oder andere Dateien
            else:
                location = "Keine Seiten- oder Tabellenblattinformation"

            formatted_docs.append(
                f"""
                    [Quelle {i}]

                    Datei:
                    {source}

                    Dateityp:
                    {file_type}

                    Fundstelle:
                    {location}

                    Inhalt:
                    {doc.page_content}
                    """
            )

        return "\n\n".join(formatted_docs)
    

    def format_history(self) -> str:
        """Returns the history formated with the roles: AI, User

        Returns:
            str: Returns history
        """        
        text = ""

        for role, msg in self.history:
            text += f"{role}: {msg}\n"

        return text



    def ask(self, question):
        """Defines the Questions-asking loop.

        Args:
            question (str): Text of the question

        Returns:
            List[str]: Answer
            List[str]: Sources of the answer
        """        
        docs = self.retriever.invoke(question)
        context = self.format_docs(docs)
        messages = self.prompt.invoke(
            {
                "history": self.format_history(),
                "context": context,
                "question": question
            }
        )

        response = self.llm.invoke(messages)
        answer = response.content

        self.history.append(("User", question))

        self.history.append(("AI", answer))

        sources = []
        for doc in docs:
            sources.append(
                {
                    "file": doc.metadata.get("source"),
                    "page": doc.metadata.get("page", 0) + 1
                }
            )

        return answer, sources, context


    def get_sources(self, documents):
        """Get sources files

        Args:
            documents (list[Documents]): List of the Documents

        Returns:
            list[Unknown]: Return of Sources as list
        """        
        unique_sources = set()

        for document in documents:
            source = document.metadata.get("source", "Unknown")

            file_type = document.metadata.get("file_type","Unknown")
            
            page = document.metadata.get("page")
            
            sheet_name = document.metadata.get("page_name")
            
            unique_sources.add(
                (
                    source,
                    file_type,
                    page,
                    sheet_name
                )
            )

        sources = []

        for (source, file_type, page, sheet_name) in sorted(unique_sources):
            item = {
                "file": source,
                "type": file_type
            }

            if page is not None:
                item["page"] = page + 1

            if sheet_name:
                item["sheet"] = sheet_name
        
            sources.append(item)

        return sources
