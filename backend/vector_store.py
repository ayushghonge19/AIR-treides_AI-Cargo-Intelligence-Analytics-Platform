"""
AIR-treides — Dual-Engine Vector Store & RAG Engine.
Supports Supabase pgvector (Primary Cloud HNSW Index) with seamless
automatic fallback to local FAISS (In-Memory / Standby Vector Store).
Handles embedding generation (all-MiniLM-L6-v2, 384 dimensions),
document chunking, ingestion, and cosine similarity search.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
import dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
import sqlalchemy
from sqlalchemy import text
from urllib.parse import quote_plus, urlparse, urlunparse

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

dotenv.load_dotenv()
logger = logging.getLogger(__name__)

# Model config: all-MiniLM-L6-v2 is ultra-fast, 384 dimensions, zero API cost
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Singleton model loader
_embedder: Optional[SentenceTransformer] = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


# ---------------------------------------------------------------------------
# Local FAISS Fallback Engine
# ---------------------------------------------------------------------------

class FAISSFallbackStore:
    """In-memory FAISS Index with Cosine Similarity (IndexFlatIP with normalized vectors)."""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
        self.docs: List[Dict[str, Any]] = []

    def add_document(self, doc_id: int, title: str, category: str, airline: str, content: str, embedding: List[float]):
        if not FAISS_AVAILABLE or self.index is None:
            return
        vec = np.array([embedding], dtype=np.float32)
        # Ensure unit normalization for Inner Product to equal Cosine Similarity
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.docs.append({
            "id": doc_id,
            "title": title,
            "category": category,
            "airline": airline,
            "content": content,
        })

    def search(self, query_embedding: List[float], top_k: int = 3, airline: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        if not FAISS_AVAILABLE or self.index is None or len(self.docs) == 0:
            return []
        
        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        k = min(len(self.docs), max(top_k * 2, 10))
        scores, indices = self.index.search(vec, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(self.docs):
                continue
            doc = self.docs[idx]
            if airline and airline.upper() != "ALL" and doc["airline"].upper() not in (airline.upper(), "ALL"):
                continue
            if category and doc["category"] != category:
                continue
            
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "airline": doc["airline"],
                "content": doc["content"],
                "similarity_score": round(float(score), 4),
            })
            if len(results) >= top_k:
                break
        return results


# Global FAISS instance
_faiss_store = FAISSFallbackStore()


def get_faiss_store() -> FAISSFallbackStore:
    return _faiss_store


# ---------------------------------------------------------------------------
# Database & pgvector Helpers
# ---------------------------------------------------------------------------

def get_safe_db_url(raw_url: str) -> str:
    """Safely URL-encodes special characters in database passwords."""
    parsed = urlparse(raw_url)
    password = parsed.password
    if password:
        safe_password = quote_plus(password)
        safe_netloc = f"{parsed.username}:{safe_password}@{parsed.hostname}"
        if parsed.port:
            safe_netloc += f":{parsed.port}"
        return urlunparse(
            (
                parsed.scheme,
                safe_netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
    return raw_url


def get_db_engine():
    supabase_url = os.getenv("SUPABASE_URL")
    if not supabase_url:
        raise ValueError("SUPABASE_URL environment variable is not set.")
    safe_url = get_safe_db_url(supabase_url)
    if safe_url.startswith("postgresql://"):
        safe_url = safe_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return sqlalchemy.create_engine(safe_url, pool_pre_ping=True)


def init_vector_table():
    """Ensure pgvector extension, cargo_guidelines table, and HNSW index exist."""
    try:
        engine = get_db_engine()
        with engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(
                text(
                    f"""
                CREATE TABLE IF NOT EXISTS cargo_guidelines (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    airline TEXT NOT NULL DEFAULT 'ALL',
                    content TEXT NOT NULL,
                    metadata_json JSONB DEFAULT '{{}}',
                    embedding vector({EMBEDDING_DIM}),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            """
                )
            )
            conn.execute(
                text(
                    """
                CREATE INDEX IF NOT EXISTS idx_cargo_guidelines_hnsw 
                ON cargo_guidelines 
                USING hnsw (embedding vector_cosine_ops);
            """
                )
            )
        logger.info("pgvector cargo_guidelines table initialized with HNSW index.")
    except Exception as exc:
        logger.warning(f"pgvector init error (FAISS fallback will be used): {exc}")


def get_embedding(text_content: str) -> List[float]:
    """Generate 384-dim normalized embedding for a text string."""
    model = get_embedder()
    embedding = model.encode(text_content, normalize_embeddings=True)
    return embedding.tolist()


def ingest_guideline(
    title: str,
    category: str,
    content: str,
    airline: str = "ALL",
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Embeds and saves a guideline chunk into Supabase pgvector and FAISS fallback."""
    embedding = get_embedding(content)
    meta_json = json.dumps(metadata or {})
    vec_str = "[" + ",".join(map(str, embedding)) + "]"
    doc_id = len(_faiss_store.docs) + 1

    # 1. Ingest to Supabase pgvector if available
    try:
        engine = get_db_engine()
        with engine.begin() as conn:
            res = conn.execute(
                text(
                    """
                    INSERT INTO cargo_guidelines (title, category, airline, content, metadata_json, embedding)
                    VALUES (:title, :category, :airline, :content, :metadata_json, CAST(:embedding AS vector))
                    RETURNING id;
                """
                ),
                {
                    "title": title,
                    "category": category,
                    "airline": airline,
                    "content": content,
                    "metadata_json": meta_json,
                    "embedding": vec_str,
                },
            )
            row = res.fetchone()
            if row:
                doc_id = row[0]
    except Exception as exc:
        logger.warning(f"Could not ingest into Supabase pgvector ({exc}); saved into FAISS fallback.")

    # 2. Always register into FAISS index as fallback
    _faiss_store.add_document(
        doc_id=doc_id,
        title=title,
        category=category,
        airline=airline,
        content=content,
        embedding=embedding,
    )
    return doc_id


def search_guidelines(
    query: str,
    top_k: int = 3,
    airline: Optional[str] = None,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Performs cosine similarity search using pgvector, falling back to FAISS if needed."""
    query_embedding = get_embedding(query)
    
    # 1. Try Supabase pgvector with HNSW index
    try:
        vec_str = "[" + ",".join(map(str, query_embedding)) + "]"
        engine = get_db_engine()

        where_clauses = []
        params: Dict[str, Any] = {"vec": vec_str, "top_k": top_k}

        if airline and airline.upper() != "ALL":
            where_clauses.append("(airline = :airline OR airline = 'ALL')")
            params["airline"] = airline

        if category:
            where_clauses.append("category = :category")
            params["category"] = category

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        sql = f"""
            SELECT 
                id,
                title,
                category,
                airline,
                content,
                1 - (embedding <=> CAST(:vec AS vector)) AS similarity_score
            FROM cargo_guidelines
            {where_sql}
            ORDER BY embedding <=> CAST(:vec AS vector)
            LIMIT :top_k;
        """

        with engine.connect() as conn:
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
            if rows:
                return [
                    {
                        "id": r[0],
                        "title": r[1],
                        "category": r[2],
                        "airline": r[3],
                        "content": r[4],
                        "similarity_score": round(float(r[5]), 4) if r[5] is not None else 0.0,
                    }
                    for r in rows
                ]
    except Exception as exc:
        logger.warning(f"pgvector query failed ({exc}). Falling back to local FAISS index.")

    # 2. Seamless Fallback: Search local FAISS index
    return _faiss_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        airline=airline,
        category=category,
    )


def get_all_guidelines_summary() -> List[Dict[str, Any]]:
    """Fetch all stored guidelines for dashboard display (with fallback)."""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT id, title, category, airline, content, created_at FROM cargo_guidelines ORDER BY id DESC LIMIT 50"
                )
            )
            return [
                {
                    "id": r[0],
                    "title": r[1],
                    "category": r[2],
                    "airline": r[3],
                    "content": r[4],
                    "created_at": str(r[5]),
                }
                for r in result.fetchall()
            ]
    except Exception:
        # Return from FAISS local memory
        return [
            {
                "id": doc["id"],
                "title": doc["title"],
                "category": doc["category"],
                "airline": doc["airline"],
                "content": doc["content"],
                "created_at": "In-Memory (FAISS)",
            }
            for doc in reversed(_faiss_store.docs)
        ]


def extract_text_from_file(file_obj, filename: str) -> str:
    """Extracts raw text from PDF, TXT, or Markdown file objects."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        import pypdf
        reader = pypdf.PdfReader(file_obj)
        pages_text = []
        for i, page in enumerate(reader.pages):
            page_content = page.extract_text() or ""
            if page_content.strip():
                pages_text.append(f"--- Page {i+1} ---\n{page_content.strip()}")
        return "\n\n".join(pages_text)
    else:
        # TXT or MD
        if hasattr(file_obj, "read"):
            content = file_obj.read()
            if isinstance(content, bytes):
                return content.decode("utf-8", errors="ignore")
            return str(content)
        return ""


def chunk_text(text_content: str, chunk_size: int = 600, overlap: int = 100) -> List[str]:
    """Splits text hierarchically into overlapping chunks."""
    if not text_content:
        return []
    paragraphs = text_content.split("\n\n")
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if len(current_chunk) + len(p_clean) < chunk_size:
            current_chunk = (current_chunk + "\n\n" + p_clean).strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(p_clean) > chunk_size:
                start = 0
                while start < len(p_clean):
                    end = start + chunk_size
                    chunks.append(p_clean[start:end])
                    start += chunk_size - overlap
                current_chunk = ""
            else:
                current_chunk = p_clean

    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def process_and_ingest_document(
    file_obj,
    filename: str,
    category: str = "General SOP",
    airline: str = "ALL",
) -> Dict[str, Any]:
    """Extracts, chunks, embeds, and stores unstructured document into pgvector & FAISS."""
    try:
        raw_text = extract_text_from_file(file_obj, filename)
        if not raw_text.strip():
            return {"success": False, "error": "No readable text could be extracted from the file."}

        chunks = chunk_text(raw_text)
        if not chunks:
            return {"success": False, "error": "File text was empty after processing."}

        ingested_ids = []
        base_title = os.path.splitext(filename)[0]

        for i, chunk in enumerate(chunks):
            chunk_title = f"{base_title} (Part {i+1}/{len(chunks)})" if len(chunks) > 1 else base_title
            doc_id = ingest_guideline(
                title=chunk_title,
                category=category,
                airline=airline,
                content=chunk,
                metadata={"source_file": filename, "chunk_index": i, "total_chunks": len(chunks)},
            )
            ingested_ids.append(doc_id)

        return {
            "success": True,
            "filename": filename,
            "chunks_count": len(chunks),
            "characters_count": len(raw_text),
            "ingested_ids": ingested_ids,
        }
    except Exception as exc:
        logger.error(f"Error processing document {filename}: {exc}")
        return {"success": False, "error": str(exc)}

