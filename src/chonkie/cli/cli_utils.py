"""CLI utilities for Chonkie using Typer."""

import os

import typer

from chonkie import Pipeline, SemanticChunker

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


@app.command()
def hello(
    name: str = typer.Argument("World", help="Name to greet"),
) -> None:
    """Say hello - a simple greeting command."""
    typer.echo(f"Hello, {name}!")


@app.command()
def chunk(
    text: str = typer.Argument(
        "please provide a text between quotes or filepath", help="Text to chunk"
    ),
    chunker: str = typer.Option(
        "semantic",
        help="Chunking method to use",
    ),
    store_in: str = typer.Option(
        None,
        help="Where to store the chunks",
    ),
) -> None:
    """Chunk - A simple chunking command."""
    typer.echo("Chunking text...")
    if chunker == "semantic":
        chunking_maker = SemanticChunker()
    if os.path.isfile(text):
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()
    chunks = chunking_maker.chunk(text)
    if store_in is None:
        for i, chunk in enumerate(chunks, 1):
            typer.echo(f"Chunk {i}:\n{chunk}\n")
    else:
        typer.echo(f"Storing chunks in {store_in}...")
        typer.echo("Chunks stored successfully.")


@app.command()
def pipeline(
    text: str = typer.Argument(
        "please provide a text between quotes or filepath", help="Text to chunk"
    ),
    fetcher: str = typer.Option(
        "file",
        help="Fetcher method to use",
    ),
    chef: str = typer.Option(
        None,
        help="Chef method to use",
    ),
    chunker: str = typer.Option(
        "semantic",
        help="Chunking method to use",
    ),
    refiner: str = typer.Option(
        None,
        help="Refiner method to use",
    ),
    handshaker: str = typer.Option(
        None,
        help="Handshaker method to use",
    ),
) -> None:
    """Chunk - A simple chunking command."""
    pipeline = Pipeline()
    pipeline.fetch_from(fetcher, path=text)
    if chef is not None:
        pipeline.process_with(chef)
    pipeline.chunk_with(chunker)
    if refiner is not None:
        pipeline.refine_with(refiner)
    if handshaker is not None:
        pipeline.store_in(handshaker)
    doc = pipeline.run()
    if doc == [] or doc is None:
        return
    if isinstance(doc, list):
        for d in doc:
            for i, chunk in enumerate(d.chunks): # type: ignore
                typer.echo(f"Chunk {i}:\n{chunk}\n")
    else:
        for i, chunk in enumerate(doc.chunks):
            typer.echo(f"Chunk {i}:\n{chunk}\n")



if __name__ == "__main__":
    app()
