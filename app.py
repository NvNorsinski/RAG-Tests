from pathlib import Path

import streamlit as st

from rag.services.rag import RagService

st.set_page_config(
    page_title="Lokaler RAG Assistent",
)

st.title("Lokaler Dokumenten-Assistent")


# RAG nur einmal laden
@st.cache_resource
def load_rag():
    return RagService()
rag = load_rag()



# Chatverlauf UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Eingabe
question = st.chat_input(
    "Frage zum Dokument..."
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Suche Dokumente..."):
            answer, sources = rag.ask(question)

        st.markdown(answer)
        st.markdown("---")
        st.markdown("### Quellen")

        for source in sources:

            file_name = Path(source["file"]).name

            text = f"📄 {file_name}"

            if "page" in source:
                text += (f" — Page "
                    f"{source['page']}")

            if "sheet" in source:
                text += (
                    f" — Sheet "
                    f"{source['sheet']}"
                )
            st.write(text)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )