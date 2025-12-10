"""CLI utilities for Chonkie using Typer."""

import os

import typer

from chonkie import SemanticChunker
from chonkie.utils import load_token
from chonkie.utils import login as login_function

# Create the main Typer app
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
) -> None:
    """chunk - A simple chunking command."""
    typer.echo("Chunking text...")
    if chunker == "semantic":
        chunking_maker = SemanticChunker()
    if os.path.isfile(text):
        with open(text, "r", encoding="utf-8") as f:
            text = f.read()
    chunks = chunking_maker.chunk(text)
    for i, chunk in enumerate(chunks, 1):
        typer.echo(f"Chunk {i}:\n{chunk}\n")


@app.command()
def login(
    api_key: str = typer.Option(
        ...,
        prompt=True,
        hide_input=True,
        help="Your API key for authentication",
    ),
) -> None:
    """Login to Chonkie CLI with your API key."""
    
    login_function(api_key)


# @app.command()
# def help_cmd(
#     ctx: typer.Context,
# ) -> None:
#     """Show help information about Chonkie CLI."""
#     if ctx.parent:
#         typer.echo(ctx.parent.get_help())


if __name__ == "__main__":
    app()
