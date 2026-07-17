from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from functools import partial
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, Field

from nemo_retriever.retriever import Retriever


MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-nemotron-embed-1b-v2")
LANCEDB_URI = os.getenv(
    "LANCEDB_URI", "/localhome/local-rkalani/nemo-retriever/lancedb"
)
TABLE_NAME = os.getenv("TABLE_NAME", "agentic-rag-documents")
EMBEDDING_ENDPOINT = os.getenv(
    "EMBEDDING_ENDPOINT", "http://127.0.0.1:8001/v1/embeddings"
)
API_TITLE = os.getenv("API_TITLE", "NeMo Retriever API")
MAX_IN_FLIGHT = int(os.getenv("MAX_IN_FLIGHT", "32"))
QUERY_TIMEOUT_SECONDS = float(os.getenv("QUERY_TIMEOUT_SECONDS", "60"))

retriever: Retriever | None = None
query_slots = asyncio.Semaphore(MAX_IN_FLIGHT)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    num_chunks: int = Field(
        default=5,
        ge=1,
        le=100,
        validation_alias=AliasChoices("num_chunks", "top_k"),
    )


class Chunk(BaseModel):
    rank: int
    text: str
    source: str | None = None
    page_number: int | None = None
    distance: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query: str
    num_chunks: int
    count: int
    chunks: list[Chunk]


@asynccontextmanager
async def lifespan(_: FastAPI):
    global retriever
    retriever = Retriever(
        run_mode="service",
        top_k=5,
        rerank=False,
        vdb_kwargs={"uri": LANCEDB_URI, "table_name": TABLE_NAME},
        embed_kwargs={
            "model_name": MODEL_NAME,
            "embed_model_name": MODEL_NAME,
            "embed_invoke_url": EMBEDDING_ENDPOINT,
            "request_timeout_s": QUERY_TIMEOUT_SECONDS,
        },
    )

    # Build the NeMo query graph and verify that the persistent embedding
    # service is ready before this worker accepts traffic.
    await asyncio.to_thread(retriever.query, "Retriever service warmup", top_k=1)
    yield
    retriever = None


app = FastAPI(
    title=API_TITLE,
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if retriever is not None else "starting",
        "model": MODEL_NAME,
        "table": TABLE_NAME,
        "embedding_endpoint": EMBEDDING_ENDPOINT,
        "max_in_flight_per_worker": MAX_IN_FLIGHT,
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is still starting")

    # NeMo sends embeddings to the always-on vLLM server. Run the synchronous
    # graph off the event loop so each API worker can serve requests in parallel.
    async with query_slots:
        try:
            hits = await asyncio.wait_for(
                asyncio.to_thread(
                    partial(retriever.query, request.query, top_k=request.num_chunks)
                ),
                timeout=QUERY_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Retriever query timed out") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Retriever query failed: {exc}") from exc

    chunks = [
        Chunk(
            rank=rank,
            text=str(hit.get("text", "")),
            source=hit.get("source"),
            page_number=hit.get("page_number"),
            distance=(float(hit["_distance"]) if hit.get("_distance") is not None else None),
            metadata=hit.get("metadata") or {},
        )
        for rank, hit in enumerate(hits, start=1)
    ]
    return QueryResponse(
        query=request.query,
        num_chunks=request.num_chunks,
        count=len(chunks),
        chunks=chunks,
    )
