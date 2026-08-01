from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import config
from services.vectorestore import VectorStoreService


class RagService:
    def __init__(self):

        self.history = []

        self.retriever = VectorStoreService().retriever()

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
        """Formats the Dokument for Output.

        Args:
            docs (list[Document]): List of returns 
        Returns:
            str: formated String
        """        
        text = ""

        for i, doc in enumerate(docs):

            page = doc.metadata["page"] + 1

            source = doc.metadata["source"]

            text += f"""
            [{i+1}]

            Datei: {source}

            Seite: {page}

            {doc.page_content}

            --------------------
            """

        return text
    

    def format_history(self):
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

        self.history.append(
            ("User", question)
        )

        self.history.append(
            ("AI", answer)
        )

        sources = []
        for doc in docs:
            sources.append(
                {
                    "file": doc.metadata.get("source"),
                    "page": doc.metadata.get("page", 0) + 1
                }
            )

        return answer, sources



 