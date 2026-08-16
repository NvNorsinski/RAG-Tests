# Testing for RAG Pipeline
This tool can answer questions on documents stored in "Dokumente".

It uses BM25 Keyword Search and Chroma Vector Search with MMR.


Change configs in config.py

Start by running the streamlit app from commandline with 


`streamlit run app.py`




Run Evaluations based on Deep Eval Package

Run specific test
python -m pytest rag/test.py -k test_rag_faithfulness -vv -s -rA

available are
test_rag_faithfulness

test_rag_relevancy

test_rag_contextual_relevancy

test_rag_hallucination



Run all tests

python -m pytest rag/test.py -k TestRAGWithDeepEval -vv -s -rA