"""CLI utilities for Chonkie using Typer."""

import os
from typing import Optional

import typer

from chonkie import Pipeline, Visualizer
from chonkie.pipeline import ComponentRegistry, ComponentType
from chonkie.types.document import Document

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

CHUNKERS = sorted(
    c.alias for c in ComponentRegistry.list_components(component_type=ComponentType.CHUNKER)
)
HANDSHAKES = sorted(
    c.alias for c in ComponentRegistry.list_components(component_type=ComponentType.HANDSHAKE)
)


@app.command()
def chunk(
    text: str = typer.Argument(..., help="Text to chunk or path to file"),
    chunker: str = typer.Option(
        "semantic",
        help=f"Chunking method to use. Options: {', '.join(CHUNKERS)}",
    ),
    handshaker: Optional[str] = typer.Option(
        None,
        help=f"Where to store the chunks. Options: {', '.join(HANDSHAKES)}",
    ),
) -> None:
    """Chunk text using a specified chunker and optionally store it."""
    typer.echo(f"Chunking with {chunker}...")

    try:
        chunker_class = ComponentRegistry.get_chunker(chunker).component_class
    except ValueError:
        typer.echo(f"Error: Unknown chunker '{chunker}'. Available: {', '.join(CHUNKERS)}")
        raise typer.Exit(code=1) from None

    chunking_maker = chunker_class()
    viz = Visualizer()
    # Get text content
    content = text
    if os.path.isfile(text):
        try:
            with open(text, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            typer.echo(f"Error reading file {text}: {e}")
            raise typer.Exit(code=1) from e

    # Chunk the text
    chunks = chunking_maker.chunk(content)

    # Handle output
    if handshaker is None:
        viz(chunks)
    else:
        try:
            handshake_class = ComponentRegistry.get_handshake(handshaker).component_class
        except ValueError:
            typer.echo(
                f"Error: Unknown handshaker '{handshaker}'. Available: {', '.join(HANDSHAKES)}"
            )
            raise typer.Exit(code=1) from None

        typer.echo(f"Storing chunks in {handshaker}...")
        try:
            handshake_instance = handshake_class()
            handshake_instance.write(chunks)
            typer.echo("Chunks stored successfully.")
        except Exception as e:
            typer.echo(f"Error storing chunks: {e}")
            raise typer.Exit(code=1) from e


@app.command()
def pipeline(
    text: Optional[str] = typer.Argument(None, help="Text to process or path to file"),
    fetcher: str = typer.Option(
        "file",
        help="Fetcher method to use (e.g., file)",
    ),
    d: Optional[str] = typer.Option(
        None,
        help="directory to process, if text is not a file",
    ),
    ext: Optional[list[str]] = typer.Option(
        None,
        help="file extensions to process, if d is specified, example ['.md', '.txt']",
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
        viz = Visualizer()
        # Configure pipeline steps

        # 1. Input Handling
        # We need to determine if we are running on:
        # - A single file (text points to file) -> use 'fetch'
        # - A directory (-d is set) -> use 'fetch'
        # - Raw text (text is string) -> pass to run()

        run_input = None

        if text is not None:
            # Check if text is a file path
            if os.path.isfile(text):
                pipe.fetch_from(fetcher, path=text)
            else:
                # Treated as direct text input
                run_input = text
        elif d is not None:
            # Check if directory exists
            if not os.path.isdir(d):
                typer.echo(f"Error: Directory '{d}' not found.")
                raise typer.Exit(code=1)
            # Pass ext only if it's not None/Empty
            pipe.fetch_from(fetcher, dir=d, ext=ext)
        else:
            typer.echo("Error: Must provide either text, a file path, or a directory via --d")
            raise typer.Exit(code=1)

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
        try:
            # If run_input is set, we pass it. If None, run() uses the fetcher step.
            doc = pipe.run(texts=run_input)
            # typer.echo(doc) # This prints the repr, which might be too verbose or ugly
        except Exception as e:
            typer.echo(f"Error running pipeline: {e}")
            raise typer.Exit(code=1) from e

        # Output results
        if handshaker:
            typer.echo(f"Pipeline completed and data stored in {handshaker}.")
            return

        if not doc:
            typer.echo("No output generated.")
            return

        docs: list[Document] = doc if isinstance(doc, list) else [doc]  # type: ignore

        for d_obj in docs:
            # Optional: print filename if available in metadata
            if d_obj.metadata and "filename" in d_obj.metadata:
                typer.echo(f"--- {d_obj.metadata['filename']} ---")

            viz(d_obj.chunks)

    except Exception as e:
        typer.echo(f"Pipeline error: {e}")
        raise typer.Exit(code=1) from e


if __name__ == "__main__":
    app()
