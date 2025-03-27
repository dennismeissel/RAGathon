# Enterprise RAG Challenge 2025

This is the solution for the [Enterprise RAG Challenge 2025](https://www.timetoact-group.at/en/insights/enterprise-rag-challenge-team-leaderboard)

# Overview

Our architecture is built with Multi-Agent LangGraph, LlamaIndex, MarkerPDF, and Llama 3.3. It extracts companies, validates input, processes each company individually, retrieves documents, evaluates feasibility, rephrases if needed, and aggregates numeric-based comparisons, ensuring accurate and contextual responses.

```
[extract_companies_node]
  --> [company_list_decider_node]
       --> if no companies => [final_na_node] -> END
       --> if one company => [company_check_node]
            --> if not found => [final_na_node] -> END
            --> if found => [clean_query_node]
                 --> [retriever_node]
                 --> [rag_evaluator_node]
                     --> if feasible => [answer_node] -> [schema_node] -> END
                     --> if not feasible => [rephrase_node] -> [retriever_node] ...
                     --> if attempts >= max_attempts and no docs => [answer_node] (with initially retrieved docs)
                     --> if attempts >= max_attempts and we do have docs => [answer_node]
                 --> if no docs found on first attempt => rephrase => ...
                 --> [final_na_node] -> END
       --> if multiple companies => [multi_company_subgraph_node]
            --> run single-company subflow for each company
            --> [comparison_aggregator_node] (numeric-based)
            --> [schema_node] -> END
```

# Getting started

## db

Holds the Vector DB created by the index process.

To clear your index completely:

```bash
rm -rf db/chroma-db
rm -rf docs-store-metadata-dir
```

## index

Responsible for document ingestion and indexing

### index with marker-pdf

1. Place files in the '/docs-to-convert' directory

2. Run the converter, then perform indexing

```bash
python index/run_md_converter.py
python index/index.py
```

## rag

RAG application for retrieving content from the vector database generated during indexing.

- Put question.json in rag folder
- Run the run_rag.py script from the root project folder
- results.json will be created in the rag folder

```bash
python rag/run_rag.py
```

To test a single query:  

```bash
python rag/test.py
```

# Credits

The solution has been created by the team **Swisscom Innovation Lab**

## Authors

- [Reto Herger](https://github.com/reherger): Parsing, indexing
- [Dennis Meissel](https://github.com/dennismeissel): Multi-agent RAG
