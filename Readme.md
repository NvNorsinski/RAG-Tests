# Testing for RAG Pipeline
This tool can answer questions. As sources it uses the documents in file "Dokumente". To do this an LLM is used.
When answering a question on the documents also the number of the page and the document where this information was found is printed out.

BM25 Keyword Search and Chroma Vector Search with MMR is used to find exact keywords in a document.
Supported document types are pdf, xlsx, xls, docx, md and txt. 

Change configs in config.py
Possible changes for example are the embedding, the LLM Model and the weight of the BM25 Keyword Search.


Start by running the streamlit app from commandline with 


`streamlit run app.py`


# Run Evaluations based on Deep Eval Package
LLM-as-a-judge evaluation is implemented. This is implemented as a Unittest to have an indicator for the correctness of the models output.

The judge model can be set in DEEPEVAL_JUDGE_MODEL in configs

Run specific test
``python -m pytest rag/test.py -k test_rag_faithfulness -vv -s -rA``

Available are:

- test_rag_faithfulness

- test_rag_relevancy

- test_rag_contextual_relevancy

- test_rag_hallucination

<br>

Run all tests with:

``python -m pytest rag/test.py -k TestRAGWithDeepEval -vv -s -rA``


# Benchmark Tokens/s

run a benchmark which measures the Tokens per second with the LLM set under LLM_MODEL in configs

start the benchmark with

``python benchmark.py``

Example output with Radeon 6700XT GPU

**Ollama LLM Benchmark**

Modell:&nbsp;gemma4:e2b

Runs:&nbsp;1

Run 1/1 ... 82.06 tok/s

ERGEBNIS

Generation:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;82.06 tok/s

Prompt processing: 746.23 tok/s

Einzelmessungen:

&nbsp;&nbsp;&nbsp;Run 1: 82.06 tok/s

Generierte Tokens: 1926

Prompt Tokens:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;81



## Requirements
All models need to be downloaded with Ollama first.