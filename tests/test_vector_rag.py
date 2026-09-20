"""
Unit tests for pgvector & FAISS Vector Store & RAG pipeline in AIR-treides.
"""

import io
import pytest
from backend.vector_store import (
    get_embedding,
    EMBEDDING_DIM,
    search_guidelines,
    ingest_guideline,
    chunk_text,
    process_and_ingest_document,
    FAISSFallbackStore,
)
from backend.nodes import search_cargo_regulations


def test_embedding_dimensions():
    """Verify embedding generation produces normalized 384-dim vector."""
    vec = get_embedding("IATA cold chain temperature standards for biopharma")
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIM
    assert all(isinstance(x, float) for x in vec)


def test_chunk_text_sliding_window():
    """Verify text chunking splits paragraphs properly with overlap."""
    sample_text = (
        "Paragraph 1: Airline hazmat guidelines for chemical storage.\n\n"
        "Paragraph 2: Packaging must be certified to Group II specifications.\n\n"
        "Paragraph 3: Inspection checklists must be verified before loading onto pallet."
    )
    chunks = chunk_text(sample_text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)


def test_process_and_ingest_txt_document():
    """Verify end-to-end document file ingestion into vector store."""
    file_content = b"Cathay Pacific Cargo SOP: Dry ice sublimation rates must not exceed 2kg per hour in cargo hold A350."
    file_obj = io.BytesIO(file_content)
    res = process_and_ingest_document(
        file_obj=file_obj,
        filename="cathay_dry_ice_sop.txt",
        category="Dangerous Goods",
        airline="Cathay Pacific",
    )
    assert res["success"] is True
    assert res["chunks_count"] >= 1
    assert len(res["ingested_ids"]) >= 1


def test_vector_search_vaccine_query():
    """Verify semantic search returns relevant cold chain document for vaccine queries."""
    results = search_guidelines("vaccine cold chain temperature", top_k=2)
    assert len(results) > 0
    top_doc = results[0]
    assert "title" in top_doc
    assert "content" in top_doc
    assert "similarity_score" in top_doc
    assert top_doc["similarity_score"] > 0.3
    assert any(term in top_doc["title"].lower() for term in ["pharma", "tcr", "temperature", "vaccine"])


def test_vector_search_dangerous_goods():
    """Verify semantic search returns lithium battery regulations."""
    results = search_guidelines("UN 3480 lithium ion battery rules on passenger plane", top_k=1)
    assert len(results) == 1
    assert "Lithium" in results[0]["title"] or "Dangerous Goods" in results[0]["category"]


def test_faiss_fallback_engine_directly():
    """Verify local FAISS index operates accurately standalone."""
    faiss_engine = FAISSFallbackStore()
    
    # Ingest mock test document
    sample_emb = get_embedding("Perishable fresh seafood must be refrigerated at 0C to 2C.")
    faiss_engine.add_document(
        doc_id=999,
        title="Lufthansa Fresh Seafood SOP",
        category="Perishables",
        airline="Lufthansa",
        content="Perishable fresh seafood must be refrigerated at 0C to 2C.",
        embedding=sample_emb,
    )
    
    query_emb = get_embedding("What temperature is needed for raw fish cargo?")
    results = faiss_engine.search(query_emb, top_k=1)
    assert len(results) == 1
    assert results[0]["title"] == "Lufthansa Fresh Seafood SOP"
    assert results[0]["similarity_score"] > 0.5


def test_search_cargo_regulations_tool():
    """Test LangChain tool execution wrapper."""
    output = search_cargo_regulations.invoke({"query": "live animal ventilation rules"})
    assert isinstance(output, str)
    assert "IATA" in output or "Animals" in output
