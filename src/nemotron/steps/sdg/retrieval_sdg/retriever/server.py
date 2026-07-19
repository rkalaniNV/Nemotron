# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FastAPI query API over the LanceDB index — wraps nemo_retriever.retriever.Retriever.

POST /query {"query": "...", "num_chunks": 5}  (top_k aliases num_chunks). Env knobs:
MODEL_NAME, LANCEDB_URI, TABLE_NAME, EMBEDDING_ENDPOINT, MAX_IN_FLIGHT, QUERY_TIMEOUT_SECONDS.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, Field

from nemo_retriever.retriever import Retriever

MODEL_NAME = os.getenv("MODEL_NAME", "nvidia/llama-nemotron-embed-1b-v2")
LANCEDB_URI = os.getenv("LANCEDB_URI", "./lancedb")
TABLE_NAME = os.getenv("TABLE_NAME", "agentic-rag-documents")
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "http://127.0.0.1:8001/v1/embeddings")
MAX_IN_FLIGHT = int(os.getenv("MAX_IN_FLIGHT", "32"))
QUERY_TIMEOUT_SECONDS = float(os.getenv("QUERY_TIMEOUT_SECONDS", "60"))

retriever: Retriever | None = None
query_slots = asyncio.Semaphore(MAX_IN_FLIGHT)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    num_chunks: int = Field(default=5, ge=1, le=100,
                            validation_alias=AliasChoices("num_chunks", "top_k"))


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
        run_mode="service", top_k=5, rerank=False,
        vdb_kwargs={"uri": LANCEDB_URI, "table_name": TABLE_NAME},
        embed_kwargs={"model_name": MODEL_NAME, "embed_model_name": MODEL_NAME,
                      "embed_invoke_url": EMBEDDING_ENDPOINT, "request_timeout_s": QUERY_TIMEOUT_SECONDS})
    # warm the query graph + verify the embedding service before accepting traffic
    await asyncio.to_thread(retriever.query, "Retriever service warmup", top_k=1)
    yield
    retriever = None


app = FastAPI(title=os.getenv("API_TITLE", "NeMo Retriever API"), version="2.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok" if retriever is not None else "starting", "model": MODEL_NAME,
            "table": TABLE_NAME, "embedding_endpoint": EMBEDDING_ENDPOINT,
            "max_in_flight_per_worker": MAX_IN_FLIGHT}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is still starting")

    # run the synchronous NeMo graph off the event loop so workers serve in parallel
    async with query_slots:
        try:
            hits = await asyncio.wait_for(
                asyncio.to_thread(retriever.query, request.query, top_k=request.num_chunks),
                timeout=QUERY_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail="Retriever query timed out") from exc
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Retriever query failed: {exc}") from exc

    chunks = [
        Chunk(rank=rank, text=str(hit.get("text", "")), source=hit.get("source"),
              page_number=hit.get("page_number"),
              distance=float(hit["_distance"]) if hit.get("_distance") is not None else None,
              metadata=hit.get("metadata") or {})
        for rank, hit in enumerate(hits, start=1)
    ]
    return QueryResponse(query=request.query, num_chunks=request.num_chunks,
                         count=len(chunks), chunks=chunks)
