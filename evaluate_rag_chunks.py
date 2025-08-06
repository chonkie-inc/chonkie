#!/usr/bin/env python3
"""RAG chunk‑quality evaluation script.

The script loads ``notebooks/data/rag.txt`` and runs a set of Chonkie
chunkers. For each chunker we compute a simple quality score (0‑10) that
takes into account:

* average chunk size vs a target (1024 tokens)
* size consistency (std/avg)
* number of chunks vs the ideal number (total tokens / target)
* processing time (penalises > 1 s)

The script prints a table with the metrics, the quality rating, and a
preview of the first two chunks for each chunker.
"""

import os, time, json, statistics
from dataclasses import dataclass
from typing import List, Dict, Any

# ---------------------------------------------------------------------------
# Import Chonkie components
# ---------------------------------------------------------------------------
from chonkie import (
    TokenChunker,
    SentenceChunker,
    RecursiveChunker,
    SemanticChunker,
    SDPMChunker,
)
from chonkie.embeddings import AutoEmbeddings

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ChunkingResult:
    """Container for a chunker run and its metrics."""

    chunker_name: str
    chunks: List[Any]
    processing_time: float
    avg_chunk_size: float
    std_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int
    total_tokens: int
    chunk_count: int
    quality: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunker_name": self.chunker_name,
            "processing_time": self.processing_time,
            "avg_chunk_size": self.avg_chunk_size,
            "std_chunk_size": self.std_chunk_size,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "total_tokens": self.total_tokens,
            "chunk_count": self.chunk_count,
            "quality": self.quality,
        }

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def quality_score(res: ChunkingResult, target: int = 1024) -> float:
    """Heuristic rating (0‑10) for chunk quality.

    Penalties (max total 10):
    * size deviation (max 5)
    * variance (max 3)
    * excess chunks (max 2)
    * processing time > 1 s (max 2)
    """
    if res.chunk_count == 0:
        return 0.0
    # 1. size deviation
    dev = abs(res.avg_chunk_size - target) / target
    penalty_dev = min(5.0, dev * 5.0)
    # 2. variance
    var = (res.std_chunk_size / res.avg_chunk_size) if res.avg_chunk_size else 0.0
    penalty_var = min(3.0, var * 3.0)
    # 3. number of chunks vs ideal
    ideal = max(1, res.total_tokens / target)
    excess = max(0.0, (res.chunk_count - ideal) / ideal
    )
    penalty_chunks = min(2.0, excess * 2.0)
    # 4. processing time
    time_pen = max(0.0, (res.processing_time - 1.0) * 0.5)
    penalty_time = min(2.0, time_pen)
    score = 10.0 - (penalty_dev + penalty_var + penalty_chunks + penalty_time)
    return max(0.0, min(10.0, score))


def run_chunker(name: str, factory, text: str) -> ChunkingResult:
    start = time.time()
    chunks = factory(text)
    elapsed = time.time() - start
    if not chunks:
        return ChunkingResult(name, [], elapsed, 0, 0, 0, 0, 0, 0, 0)
    sizes = [c.token_count for c in chunks]
    total = sum(sizes)
    avg = statistics.mean(sizes)
    std = statistics.stdev(sizes) if len(sizes) > 1 else 0.0
    result = ChunkingResult(
        chunker_name=name,
        chunks=chunks,
        processing_time=elapsed,
        avg_chunk_size=avg,
        std_chunk_size=std,
        min_chunk_size=min(sizes),
        max_chunk_size=max(sizes),
        total_tokens=total,
        chunk_count=len(chunks),
    )
    result.quality = quality_score(result)
    return result

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    file_path = "./notebooks/data/rag.txt"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    text = load_text(file_path)

    # Load embedding model once for semantic chunkers
    embedding = AutoEmbeddings.get_embeddings("sentence-transformers/all-MiniLM-L6-v2")

    factories = {
        "TokenChunker": lambda txt: TokenChunker(tokenizer="character", chunk_size=1024, chunk_overlap=128)(txt),
        "SentenceChunker": lambda txt: SentenceChunker(tokenizer_or_token_counter="character", chunk_size=1024, chunk_overlap=128, min_sentences_per_chunk=1)(txt),
        "RecursiveChunker": lambda txt: RecursiveChunker(tokenizer_or_token_counter="character", chunk_size=1024, min_characters_per_chunk=24)(txt),
        "SemanticChunker": lambda txt: SemanticChunker(embedding_model=embedding, chunk_size=1024, threshold=0.5, min_sentences=1)(txt),
        "SDPMChunker": lambda txt: SDPMChunker(embedding_model=embedding, chunk_size=1024, threshold=0.5, min_sentences=1)(txt),
    }

    results: List[ChunkingResult] = []
    for name, factory in factories.items():
        try:
            results.append(run_chunker(name, factory, text))
        except Exception as e:
            print(f"{name} failed: {e}")

    # Print report with quality scores
    print("\n" + "=" * 80)
    print("CHUNKING EVALUATION WITH QUALITY SCORE")
    print("=" * 80)
    header = f"{'Chunker':<20} {'Chunks':<8} {'Avg':<8} {'Std':<8} {'Min':<5} {'Max':<5} {'Time(s)':<8} {'Score':<5}"
    print(header)
    print("-" * 80)
    for r in results:
        print(f"{r.chunker_name:<20} {r.chunk_count:<8} {r.avg_chunk_size:<8.1f} {r.std_chunk_size:<8.1f} {r.min_chunk_size:<5} {r.max_chunk_size:<5} {r.processing_time:<8.2f} {r.quality:<5.1f}")
    print("=" * 80)

    # Show sample chunks (first 2) for each chunker
    for r in results:
        print(f"\n--- {r.chunker_name} sample chunks (first 2) ---")
        for i, chunk in enumerate(r.chunks[:2]):
            txt = str(chunk)[:200].replace('\n', ' ')
            print(f"Chunk {i+1}: {txt} ...")

    # Save JSON for later analysis
    with open("chunking_quality.json", "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print("\nResults saved to chunking_quality.json")

if __name__ == "__main__":
    main()
