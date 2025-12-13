"""CLI utilities for Chonkie using Typer."""

import os
from typing import Optional

import typer

from chonkie import (
    ChromaHandshake,
    CodeChunker,
    ElasticHandshake,
    LateChunker,
    MilvusHandshake,
    MongoDBHandshake,
    NeuralChunker,
    PgvectorHandshake,
    PineconeHandshake,
    Pipeline,
    QdrantHandshake,
    RecursiveChunker,
    SemanticChunker,
    SentenceChunker,
    SlumberChunker,
    TableChunker,
    TokenChunker,
    TurbopufferHandshake,
    WeaviateHandshake,
)

# from chonkie.utils import login as login_function

# @app.command()
# def login(
#     api_key: str = typer.Option(
#         ...,
#         prompt=True,
#         hide_input=True,
#         help="Your API key for authentication",
#     ),
# ) -> None:
#     """Login to Chonkie CLI with your API key."""
#     login_function(api_key)


app = typer.Typer(
    name="chonkie",
    help=">🦛 CHONK your texts with Chonkie",
    add_completion=True,
)

CHUNKER_MAPPING = {
    "semantic": SemanticChunker,
    "recursive": RecursiveChunker,
    "token": TokenChunker,
    "sentence": SentenceChunker,
    "code": CodeChunker,
    "late": LateChunker,
    "neural": NeuralChunker,
    "slumber": SlumberChunker,
    "table": TableChunker,
}

HANDSHAKE_MAPPING = {
    "chroma": ChromaHandshake,
    "elastic": ElasticHandshake,
    "milvus": MilvusHandshake,
    "mongodb": MongoDBHandshake,
    "pgvector": PgvectorHandshake,
    "pinecone": PineconeHandshake,
    "qdrant": QdrantHandshake,
    "turbopuffer": TurbopufferHandshake,
    "weaviate": WeaviateHandshake,
}


@app.command()
def chunk(
    text: str = typer.Argument(..., help="Text to chunk or path to file"),
    chunker: str = typer.Option(
        "semantic",
        help=f"Chunking method to use. Options: {', '.join(CHUNKER_MAPPING.keys())}",
    ),
    handshaker: Optional[str] = typer.Option(
        None,
        help=f"Where to store the chunks. Options: {', '.join(HANDSHAKE_MAPPING.keys())}",
    ),
) -> None:
    """Chunk text using a specified chunker and optionally store it."""
    typer.echo(f"Chunking with {chunker}...")

    # Select chunker
    if chunker not in CHUNKER_MAPPING:
        typer.echo(
            f"Error: Unknown chunker '{chunker}'. Available: {', '.join(CHUNKER_MAPPING.keys())}"
        )
        raise typer.Exit(code=1)

    chunker_class = CHUNKER_MAPPING[chunker]
    chunking_maker = chunker_class()

    # Get text content
    content = text
    if os.path.isfile(text):
        try:
            with open(text, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            typer.echo(f"Error reading file {text}: {e}")
            raise typer.Exit(code=1)

    # Chunk the text
    chunks = chunking_maker.chunk(content)

    # Handle output
    if handshaker is None:
        for i, chunk in enumerate(chunks):
            typer.echo(f"Chunk {i}:\n{chunk}\n")
    else:
        if handshaker not in HANDSHAKE_MAPPING:
            typer.echo(
                f"Error: Unknown handshaker '{handshaker}'. Available: {', '.join(HANDSHAKE_MAPPING.keys())}"
            )
            raise typer.Exit(code=1)

        typer.echo(f"Storing chunks in {handshaker}...")
        try:
            handshake_class = HANDSHAKE_MAPPING[handshaker]
            handshake_instance = handshake_class()
            handshake_instance.write(chunks)
            typer.echo("Chunks stored successfully.")
        except Exception as e:
            typer.echo(f"Error storing chunks: {e}")
            raise typer.Exit(code=1)


@app.command()
def pipeline(
    text: str = typer.Argument(..., help="Text to process or path to file/directory"),
    fetcher: str = typer.Option(
        "file",
        help="Fetcher method to use (e.g., file)",
    ),
    chef: Optional[str] = typer.Option(
        None,
        help="Chef method to use (e.g., text, markdown)",
    ),
    chunker: str = typer.Option(
        "semantic",
        help="Chunking method to use",
    ),
    refiner: Optional[str] = typer.Option(
        None,
        help="Refiner method to use",
    ),
    handshaker: Optional[str] = typer.Option(
        None,
        help="Handshaker method to use",
    ),
) -> None:
    """Run a processing pipeline on text or files."""
    try:
        pipe = Pipeline()

        # Configure pipeline steps
        # 1. Fetcher / Input
        if os.path.exists(text):
            pipe.fetch_from(fetcher, path=text)
        else:
            # If text is not a file, we treat it as direct input
            pass

        # 2. Chef
        if chef is not None:
            pipe.process_with(chef)

        # 3. Chunker
        pipe.chunk_with(chunker)

        # 4. Refiner
        if refiner is not None:
            pipe.refine_with(refiner)

        # 5. Handshaker
        if handshaker is not None:
            pipe.store_in(handshaker)

        # Run pipeline
        typer.echo("Running pipeline...")
        if os.path.exists(text):
            doc = pipe.run()
        else:
            doc = pipe.run(texts=text)

        # Output results
        if handshaker:
            typer.echo(f"Pipeline completed and data stored in {handshaker}.")
            return

        if not doc:
            typer.echo("No output generated.")
            return
        
        # i need to review this later
        docs = doc if isinstance(doc, list) else [doc]

        for d in docs:
            for i, chunk in enumerate(d.chunks):
                typer.echo(f"Chunk {i}:\n{chunk.text}\n")

    except Exception as e:
        typer.echo(f"Pipeline error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
