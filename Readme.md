# Testing for RAG Pipeline
This tool can answer questions on documents stored in "Dokumente".

It uses BM25 Keyword Search and Chroma Vector Search with MMR.


Change configs in config.py
Possible Changes are the embedding and the LLM Model.
All the models need to be downloaded in Ollama.



Start by running the streamlit app from commandline with 


`streamlit run app.py`




# Run Evaluations based on Deep Eval Package
LLM-as-a-judge evaluation is implemented.

The judge model can be set in DEEPEVAL_JUDGE_MODEL in configs

Run specific test
``python -m pytest rag/test.py -k test_rag_faithfulness -vv -s -rA``

Available are:

test_rag_faithfulness

test_rag_relevancy

test_rag_contextual_relevancy

test_rag_hallucination



Run all tests

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



-------------------------------------------
All models need to be downloaded with Ollama first.