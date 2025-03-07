# rag.py

# [extract_companies_node]
#   --> [company_list_decider_node]
#        --> if no companies => [final_na_node] -> END
#        --> if one company => [company_check_node]
#             --> if not found => [final_na_node] -> END
#             --> if found => [clean_query_node]
#                  --> [retriever_node]
#                  --> [rag_evaluator_node]
#                      --> if feasible => [answer_node] -> [schema_node] -> END
#                      --> if not feasible => [rephrase_node] -> [retriever_node] ...
#                      --> if attempts >= max_attempts and no docs => [answer_node] (with initially retrieved docs)
#                      --> if attempts >= max_attempts and we do have docs => [answer_node]
#                  --> if no docs found on first attempt => rephrase => ...
#                  --> [final_na_node] -> END
#        --> if multiple companies => [multi_company_subgraph_node]
#             --> run single-company subflow for each company
#             --> [comparison_aggregator_node] (numeric-based)
#             --> [schema_node] -> END

import sys
import os
import logging
import re
import json
from dotenv import load_dotenv
from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from db.vector_store import get_vector_store
from index.settings import init_settings
from llama_index.core import Settings
from llama_index.core.indices import VectorStoreIndex
from llama_index.core.vector_stores.types import (
    MetadataFilters,
    MetadataFilter,
    FilterOperator,
)

##############################################################################
# Setup Logging
##############################################################################
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

##############################################################################
# 1) Environment/LLM Setup
##############################################################################
load_dotenv()

BASE_URL = os.getenv("BASE_URL", "")
API_KEY = os.getenv("API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "")

llm = ChatOpenAI(
    base_url=BASE_URL,
    api_key=API_KEY,
    model=LLM_MODEL,
    temperature=0,
)

init_settings()

##############################################################################
# 2) Vector Store / Index Setup
##############################################################################
index = VectorStoreIndex.from_vector_store(get_vector_store())

##############################################################################
# 3) Define the WorkflowState
##############################################################################
class WorkflowState(TypedDict):
    query: Annotated[str, "query"]
    corrected_query: Annotated[str, "corrected_query"]
    company_name: Annotated[str, "company_name"]
    company_list: Annotated[List[str], "company_list"]  # supports multi-company
    schema: Annotated[str, "schema"]  # e.g. 'number', 'boolean', etc.

    retrieved_docs: Annotated[List[Dict[str, Any]], "retrieved_docs"]
    retrieved_initially: Annotated[List[Dict[str, Any]], "retrieved_initially"]
    answer: Annotated[str, "answer"]
    llm_answer: Annotated[str, "llm_answer"]
    attempts: Annotated[int, "attempts"]
    next_step: Annotated[str, "next_step"]
    top_k: Annotated[int, "top_k"]
    max_attempts: Annotated[int, "max_attempts"]
    references: Annotated[List[Dict[str, str]], "references"]

    # For multi-company queries:
    # Each item: { "company":..., "answer":..., "references":[...] }
    multi_answers: Annotated[List[Dict[str, Any]], "multi_answers"]


##############################################################################
# 4) Define Node Functions
##############################################################################
def remove_reasoning_part(answer: str) -> str:
    """
    Removes the reasoning part within <think> tags, if present.
    """
    return re.sub(r".*?</think>\n?", "", answer, flags=re.DOTALL).strip()


def extract_companies_node(state: WorkflowState) -> dict:
    """
    Extract ALL company names from the user's query.
    Returns them as a list in state["company_list"].
    If none are found, it will be an empty list.
    """
    user_query = state["query"]
    logger.info(f"Extracting companies from user query: {user_query}")

    prompt = f"""
Extract all company names from the text below.
Return ONLY valid JSON containing a list of strings.
If no companies, return [].

Text:
{user_query}

JSON:
"""
    response = llm.invoke(prompt).content.strip()
    try:
        cleaned = remove_reasoning_part(response)
        companies = json.loads(cleaned)
        if not isinstance(companies, list):
            companies = []
    except:
        companies = []

    logger.info(f"extract_companies_node => found companies: {companies}")
    return {"company_list": companies}


def company_list_decider_node(state: WorkflowState) -> dict:
    """
    Decide what path to take based on the number of companies found.
    - 0 => final_na_node
    - 1 => fill state["company_name"] and go single-company route
    - >1 => go multi-company route
    """
    companies = state["company_list"]
    if not companies:
        return {"next_step": "final_na_node"}
    elif len(companies) == 1:
        return {"company_name": companies[0], "next_step": "company_check_node"}
    else:
        return {"next_step": "multi_company_subgraph_node"}


def company_check_node(state: WorkflowState) -> dict:
    """
    1) If company_name is empty => final_na_node
    2) Otherwise, check if we have docs for that company.
       If none => final_na_node
       If yes => go to clean_query_node
    """
    company_name = state["company_name"]
    if not company_name:
        logger.info("No company_name => final_na_node")
        return {"next_step": "final_na_node"}

    # Quick retrieval to see if we have docs
    filters = MetadataFilters(
        filters=[MetadataFilter(key="company_name", operator=FilterOperator.EQ, value=company_name)]
    )
    checker_retriever = index.as_retriever(similarity_top_k=1, filters=filters)
    checker_docs = checker_retriever.retrieve("placeholder")

    if not checker_docs:
        logger.info(f"No docs for {company_name} => final_na_node")
        return {"next_step": "final_na_node"}
    return {"next_step": "clean_query_node"}


def clean_query_node(state: WorkflowState) -> dict:
    """
    Removes references to the specific company, 'annual report', or 'If data ... N/A'
    from state["query"], returning corrected_query.
    """
    user_query = state["query"]
    company_name = state["company_name"]
    logger.info(f"clean_query_node => original user_query: {user_query}")

    prompt = f"""
Rewrite the user query by removing references to the company "{company_name}",
the words "annual report", and the phrase "If data is not available, return 'N/A'".
Return only the cleaned version, nothing else.

User query:
{user_query}

Cleaned version:
"""
    answer = llm.invoke(prompt).content.strip()
    answer = remove_reasoning_part(answer)
    logger.info(f"clean_query_node => cleaned: {answer}")
    return {"corrected_query": answer}


def retriever_node(state: WorkflowState) -> dict:
    """
    Uses corrected_query + company_name => retrieve docs from vector store.
    If attempts=0, store as retrieved_initially.
    """
    cquery = state["corrected_query"]
    top_k = state["top_k"]
    attempts = state["attempts"]
    logger.info(f"retriever_node => attempts={attempts}, cquery={cquery}")

    # Filter to the single company
    filters = MetadataFilters(
        filters=[
            MetadataFilter(key="company_name", operator=FilterOperator.EQ, value=state["company_name"])
        ]
    )
    retriever = index.as_retriever(similarity_top_k=top_k, filters=filters)

    docs = retriever.retrieve(cquery)
    if not docs:
        logger.warning("retriever_node => no docs retrieved.")
        docs_with_meta = []
    else:
        docs_with_meta = []
        for d in docs:
            doc_text = d.node.text
            doc_meta = d.node.extra_info or {}
            docs_with_meta.append({"text": doc_text, "metadata": doc_meta})

    result = {"retrieved_docs": docs_with_meta}
    if attempts == 0:
        result["retrieved_initially"] = docs_with_meta

    return result


def rag_evaluator_node(state: WorkflowState) -> dict:
    """
    If we have docs, ask LLM "Can we answer?" => yes => answer_node, else => rephrase or final.
    If no docs => rephrase if attempts < max_attempts else => answer_node w/ initial docs.
    """
    attempts = state["attempts"] + 1
    max_attempts = state["max_attempts"]
    docs = state["retrieved_docs"]
    cquery = state["corrected_query"]

    logger.info(f"rag_evaluator_node => attempt {attempts}/{max_attempts}")

    if not docs:
        logger.warning("No docs. Possibly rephrase or final answer with initial docs.")
        if attempts >= max_attempts:
            logger.info("Max attempts => use initial docs => answer_node")
            return {
                "attempts": attempts,
                "next_step": "answer_node",
                "retrieved_docs": state["retrieved_initially"],
            }
        else:
            return {"attempts": attempts, "next_step": "rephrase_node"}

    # If we do have docs, see if we can produce an answer
    doc_context = "\n".join([f"- {d['text']}" for d in docs])
    prompt = f"""
We have a user question: {cquery}
We have these documents:
{doc_context}

Answer only 'yes' or 'no':
Can we produce a confident answer with this context?
"""
    eval_resp = remove_reasoning_part(llm.invoke(prompt).content.lower().strip())

    if "yes" in eval_resp and "no" not in eval_resp:
        return {"attempts": attempts, "next_step": "answer_node"}
    else:
        if attempts >= max_attempts:
            logger.info("LLM says 'no', but we hit max attempts => answer anyway w/ initial docs.")
            return {
                "attempts": attempts,
                "next_step": "answer_node",
                "retrieved_docs": state["retrieved_initially"],
            }
        else:
            return {"attempts": attempts, "next_step": "rephrase_node"}


def rephrase_node(state: WorkflowState) -> dict:
    """
    Rephrase corrected_query => hopefully better retrieval next pass.
    """
    old_q = state["corrected_query"]
    logger.info(f"rephrase_node => old: {old_q}")

    prompt = f"""
Rephrase (or expand) this query to improve results.
Return only the new phrasing.

Original:
{old_q}

New:
"""
    new_q = remove_reasoning_part(llm.invoke(prompt).content.strip())
    logger.info(f"rephrase_node => new: {new_q}")
    return {"corrected_query": new_q}


def answer_node(state: WorkflowState) -> dict:
    """
    Build a final answer from retrieved_docs, referencing them with bracket notation.
    """
    question = state["query"]
    docs = state["retrieved_docs"] or []

    # Build context
    lines = []
    for i, doc in enumerate(docs):
        meta = doc["metadata"]
        fname_full = meta.get("file_name", "unknown_file")
        page = meta.get("page", "unknown_page")
        fname_noext = os.path.splitext(fname_full)[0]

        lines.append(
            f"Reference {i+1}:\n[REF: {fname_noext}, PAGE: {page}]\nContent:\n{doc['text']}\n"
        )
    context_str = "\n".join(lines)

    prompt = f"""
You must answer the question using ONLY the context below.
If not enough detail, return 'N/A'.
Cite references in format [REF: <filename>, PAGE: <page>].

Question: {question}

Context:
{context_str}

Answer (with references if needed):
"""
    raw_ans = remove_reasoning_part(llm.invoke(prompt).content.strip())

    # Extract references
    pattern = r"\[REF:\s*([^,]+),\s*PAGE:\s*([^\]]+)\]"
    matches = re.findall(pattern, raw_ans)
    refs = []
    for (fname, pg) in matches:
        fname = fname.strip()
        pg = pg.strip()
        if fname.endswith(".pdf"):
            fname = fname[:-4]
        refs.append({"filename": fname, "page": pg})

    return {"answer": raw_ans, "llm_answer": raw_ans, "references": refs}


def schema_node(state: WorkflowState) -> dict:
    """
    Convert final free-text answer => requested schema (number, boolean, etc.).
    If parse fails, 'N/A'.

    For 'name', we want the EXACT company name from the final answer
    (including suffixes like "Inc.", "plc", etc.), with no extra text like
    "applying rule 3" or disclaimers.
    """
    sc = state["schema"]
    ans = state["answer"]

    prompt = f"""
We have an answer: {ans}
We want to convert it into the schema: {sc}.

Rules:
- If 'number', find the numeric value or return 'N/A' if not found.
- If 'boolean', must be yes/no/true/false, or 'N/A'.
- If 'name':
  1) Return the exact name from the answer, including suffixes ("Inc.", "plc", etc.).
  2) Remove any extra disclaimers like "Applying rule..." or "extract only the name:".
  3) If there's a newline, keep only what's before it.
  4) If you cannot find a name, return 'N/A'.
- If 'names', extract a list of valid names or 'N/A' if not found.
Otherwise, do your best or return 'N/A'.

Return only the final text that matches the requested schema (no extra explanations).
"""

    converted = remove_reasoning_part(llm.invoke(prompt).content.strip())
    return {"answer": converted}



def final_na_node(state: WorkflowState) -> dict:
    logger.info("final_na_node => answer=N/A")
    return {"answer": "N/A"}


# ---------------------------------------------------------------------------
# MULTI-COMPANY LOGIC
# ---------------------------------------------------------------------------
def multi_company_subgraph_node(state: WorkflowState) -> dict:
    """
    If multiple companies, we rewrite the question for each company and run single-company subflow.
    Store partial answers in state["multi_answers"].
    """
    companies = state["company_list"]
    logger.info(f"multi_company_subgraph_node => companies: {companies}")

    results = []
    original_question = state["query"]

    for comp in companies:
        single_query = build_single_company_subquery(original_question, comp)

        sub_state = dict(state)
        sub_state["query"] = single_query
        sub_state["corrected_query"] = single_query
        sub_state["company_name"] = comp
        sub_state["attempts"] = 0
        sub_state["retrieved_docs"] = []
        sub_state["retrieved_initially"] = []

        partial_ans = run_single_company_subflow(sub_state)
        logger.info(f"Result for {comp}: {partial_ans}")
        results.append({
            "company": comp,
            "answer": partial_ans["answer"],
            "references": partial_ans["references"],
        })

    logger.info(f"multi_company_subgraph_node => got partial results: {results}")
    return {"multi_answers": results}


def build_single_company_subquery(orig_question: str, comp: str) -> str:
    """
    Convert multi-company question into single-company question for 'comp'.
    We'll let LLM do rewriting, then remove reasoning text.
    """
    prompt = f"""
We have a multi-company question:
\"\"\"{orig_question}\"\"\"

Rewrite it so it only requests data about the single company: {comp}.
Focus on the same metric or details, specifically for {comp}.
If data is not found, mention 'N/A'.

Output only the single-company question text, nothing else.
"""
    single_q = llm.invoke(prompt).content.strip()
    single_q = remove_reasoning_part(single_q)
    logger.info(f"build_single_company_subquery => for {comp}: {single_q}")
    return single_q


def run_single_company_subflow(s: WorkflowState) -> Dict[str, Any]:
    """
    Single-company chain: check -> clean -> retrieve -> possibly rephrase -> answer.
    We'll do minimal attempts (just one pass).
    """
    updated = company_check_node(s)
    s.update(updated)
    if s.get("next_step") == "final_na_node":
        return {"answer": "N/A", "references": []}

    updated = clean_query_node(s)
    s.update(updated)

    updated = retriever_node(s)
    s.update(updated)

    docs = s["retrieved_docs"]
    if not docs:
        return {"answer": "N/A", "references": []}

    updated = answer_node(s)
    s.update(updated)
    return {"answer": s["answer"], "references": s["references"]}


def comparison_aggregator_node(state: WorkflowState) -> dict:
    """
    Attempt a numeric parse of net income from each partial answer,
    pick the smallest. Then collect references from *all* partial answers,
    so the final references array includes everything discovered.
    """
    multi_answers = state.get("multi_answers", [])
    logger.info(f"comparison_aggregator_node => multi_answers: {multi_answers}")

    if not multi_answers:
        logger.info("No multi_answers => returning N/A")
        return {"answer": "N/A", "llm_answer": "N/A", "references": []}

    best_company = None
    best_value = None

    # We'll collect references from *all* subflows
    # then unify them into a single list (remove duplicates).
    all_refs = []

    def parse_finance_string(txt: str) -> float:
        """
        Attempt a basic parse for 'xxxxx million' or 'xxxxx thousand' or plain number.
        e.g. '30,126 thousand EUR' => 30126000.0
        """
        raw = txt.lower()
        # remove ' eur'
        raw = raw.replace(" eur", "")
        # find a pattern: '2,717 million'
        pat = r"([\-\d,\.]+)\s*(million|thousand)?"
        match = re.search(pat, raw)
        if not match:
            raise ValueError("No numeric pattern found in: " + txt)
        number_str = match.group(1).replace(",", "")
        scale = match.group(2)
        val = float(number_str)
        if scale == "thousand":
            val *= 1e3
        elif scale == "million":
            val *= 1e6
        return val

    for entry in multi_answers:
        # always accumulate references
        for ref in entry["references"]:
            all_refs.append(ref)

        txt = entry["answer"]
        comp = entry["company"]
        try:
            parsed = parse_finance_string(txt)
            logger.info(f"Parsed for {comp}: {parsed}")
            if best_value is None or parsed < best_value:
                best_value = parsed
                best_company = comp
        except Exception as e:
            logger.info(f"Skipping {comp}, parse error: {e}")

    if not best_company:
        logger.info("No numeric parse => returning N/A with all references.")
        # unify the references
        deduped_refs = _deduplicate_references(all_refs)
        return {"answer": "N/A", "llm_answer": "N/A", "references": deduped_refs}

    logger.info(f"comparison_aggregator_node => chosen: {best_company}")
    # unify references from all subflows
    deduped_refs = _deduplicate_references(all_refs)
    return {
        "answer": best_company,
        "llm_answer": best_company,
        "references": deduped_refs
    }

def _deduplicate_references(refs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Utility to remove duplicates from a list of references,
    so we don't store repeated {filename, page}.
    """
    seen = set()
    out = []
    for r in refs:
        # form a tuple for dedup
        key = (r.get("filename",""), r.get("page",""))
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


##############################################################################
# 5) Build the Base StateGraph
##############################################################################
workflow = StateGraph(WorkflowState, config_schema={"recursion_limit": 25})

workflow.add_node("extract_companies_node", extract_companies_node)
workflow.add_node("company_list_decider_node", company_list_decider_node)
workflow.add_node("company_check_node", company_check_node)
workflow.add_node("clean_query_node", clean_query_node)
workflow.add_node("retriever_node", retriever_node)
workflow.add_node("rag_evaluator_node", rag_evaluator_node)
workflow.add_node("rephrase_node", rephrase_node)
workflow.add_node("answer_node", answer_node)
workflow.add_node("schema_node", schema_node)
workflow.add_node("final_na_node", final_na_node)

workflow.add_node("multi_company_subgraph_node", multi_company_subgraph_node)
workflow.add_node("comparison_aggregator_node", comparison_aggregator_node)

workflow.set_entry_point("extract_companies_node")

workflow.add_edge("extract_companies_node", "company_list_decider_node")
workflow.add_conditional_edges(
    "company_list_decider_node",
    lambda s: s["next_step"],
    {
        "final_na_node": "final_na_node",
        "company_check_node": "company_check_node",
        "multi_company_subgraph_node": "multi_company_subgraph_node",
    }
)

# Single-company path
workflow.add_conditional_edges(
    "company_check_node",
    lambda s: s["next_step"],
    {
        "final_na_node": "final_na_node",
        "clean_query_node": "clean_query_node"
    }
)
workflow.add_edge("clean_query_node", "retriever_node")
workflow.add_edge("retriever_node", "rag_evaluator_node")
workflow.add_conditional_edges(
    "rag_evaluator_node",
    lambda s: s["next_step"],
    {
        "answer_node": "answer_node",
        "rephrase_node": "rephrase_node",
        "final_na_node": "final_na_node",
    }
)
workflow.add_edge("rephrase_node", "retriever_node")
workflow.add_edge("answer_node", "schema_node")
workflow.add_edge("schema_node", END)
workflow.add_edge("final_na_node", END)

# Multi-company path
workflow.add_edge("multi_company_subgraph_node", "comparison_aggregator_node")
workflow.add_edge("comparison_aggregator_node", "schema_node")

##############################################################################
# 6) get_rag_answer
##############################################################################
def get_rag_answer(
    question: str,
    schema: str,
    top_k: int = int(os.getenv("TOP_K", 20)),
    max_attempts: int = 5
) -> Dict[str, Any]:
    """
    Main RAG entry point.
    Multi-company logic:
      - Each company is run through single-company subflow => partial answers + references.
      - We attempt to parse numeric net income from each partial answer, pick the smallest.
      - That 'winning' references get attached. If no parse => 'N/A'.
    """
    initial_state: WorkflowState = {
        "query": question,
        "corrected_query": question,
        "company_name": "",
        "company_list": [],
        "schema": schema,
        "retrieved_docs": [],
        "retrieved_initially": [],
        "answer": "",
        "llm_answer": "",
        "attempts": 0,
        "next_step": "",
        "top_k": top_k,
        "max_attempts": max_attempts,
        "references": [],
        "multi_answers": []
    }

    final_state = workflow.compile().invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "references": final_state.get("references", []),
        "llm_answer": final_state.get("llm_answer", "")
    }
