"""CLI utilities for Chonkie using Typer."""

import os
import sys
from typing import Any, Optional

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


def parse_params(param_list: list[str] | None) -> dict[str, Any]:
    """Parse a list of key=value strings into a dictionary.

    Args:
        param_list: List of strings in format "key=value" or just "key" (for boolean flags)

    Returns:
        Dictionary of parsed parameters with type conversion

    Examples:
        >>> parse_params(["chunk_size=512", "threshold=0.8", "verbose"])
        {'chunk_size': 512, 'threshold': 0.8, 'verbose': True}

    """
    if not param_list:
        return {}

    params: dict[str, Any] = {}
    for param in param_list:
        if "=" in param:
            key, value = param.split("=", 1)
            key = key.strip()
            value = value.strip()

            # Try to convert to appropriate type
            if value.lower() == "true":
                params[key] = True
            elif value.lower() == "false":
                params[key] = False
            elif value.lower() == "none" or value.lower() == "null":
                params[key] = None
            else:
                # Try float first, then convert to int if appropriate, or keep as string
                try:
                    # Try float first (handles both floats and ints in scientific notation)
                    float_val = float(value)
                    # If it's a whole number and no decimal point in original, keep as int
                    if "." not in value and "e" not in value.lower() and float_val.is_integer():
                        params[key] = int(float_val)
                    else:
                        params[key] = float_val
                except ValueError:
                    # Keep as string
                    params[key] = value
        else:
            # Boolean flag (no =value)
            params[param.strip()] = True

    return params


def merge_params(explicit_params: dict[str, Any], parsed_params: dict[str, Any]) -> dict[str, Any]:
    """Merge explicit parameters with parsed parameters, with explicit taking precedence.

    Args:
        explicit_params: Parameters from explicit CLI options
        parsed_params: Parameters from parsed key=value strings

    Returns:
        Merged dictionary

    """
    return dict(
       parsed_params,
       **{key: value for key, value in explicit_params.items() if value is not None},
    )


@app.command()
def chunk(
    text: str = typer.Argument(..., help="Text to chunk or path to file"),
    chunker: str = typer.Option(
        "semantic",
        help=f"Chunking method to use. Options: {', '.join(CHUNKERS)}",
    ),
    chunk_size: Optional[int] = typer.Option(
        None,
        help="Maximum number of tokens per chunk",
    ),
    chunk_overlap: Optional[int] = typer.Option(
        None,
        help="Number of tokens to overlap between chunks",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        help="Threshold for semantic similarity (0-1)",
    ),
    chunker_params: Optional[list[str]] = typer.Option(
        None,
        help="Additional parameters for the chunker as key=value pairs (e.g., --chunker-params tokenizer=gpt2 min_characters_per_chunk=50)",
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
        raise typer.Exit(code=1)

    # Parse and merge parameters
    explicit_params = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "threshold": threshold,
    }
    parsed_params = parse_params(chunker_params)
    chunker_kwargs = merge_params(explicit_params, parsed_params)

    # Create chunker instance with parameters
    try:
        chunking_maker = chunker_class(**chunker_kwargs)
    except Exception as e:
        typer.echo(f"Error initializing chunker with parameters: {e}")
        raise typer.Exit(code=1)

    viz = Visualizer()
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
        try:
            # Set UTF-8 encoding for stdout on Windows if possible
            if sys.platform == "win32":
                try:
                    sys.stdout.reconfigure(encoding="utf-8")
                except (AttributeError, ValueError):
                    pass  # Python < 3.7 or reconfigure not available
            viz(chunks)
        except (UnicodeEncodeError, UnicodeDecodeError, BrokenPipeError) as e:
            # Fallback for Windows console encoding issues
            try:
                if chunks:
                    typer.echo(f"Chunked into {len(chunks)} chunks:")
                    for i, chunk in enumerate(chunks, 1):
                        chunk_text = (
                            getattr(chunk, "text", "")[:200] if hasattr(chunk, "text") else ""
                        )
                        token_count = (
                            getattr(chunk, "token_count", 0)
                            if hasattr(chunk, "token_count")
                            else 0
                        )
                        typer.echo(f"\n--- Chunk {i} ({token_count} tokens) ---")
                        # Truncate and escape problematic characters for display
                        preview = chunk_text.encode("ascii", errors="replace").decode("ascii")
                        typer.echo(preview + ("..." if len(chunk_text) > 200 else ""))
                else:
                    typer.echo("No chunks to display (encoding error occurred)")
            except Exception as fallback_error:
                # If fallback also fails, show minimal error info without masking original
                typer.echo(
                    f"Encoding error ({type(e).__name__}) occurred, and fallback display also failed ({type(fallback_error).__name__})"
                )
                typer.echo(f"Original error: {e}")
                typer.echo(f"Fallback error: {fallback_error}")
        except Exception as e:
            # Catch any other visualization errors and provide basic output
            try:
                if chunks:
                    typer.echo(
                        f"Chunked into {len(chunks)} chunks (visualization error: {type(e).__name__})"
                    )
                    for i, chunk in enumerate(chunks, 1):
                        chunk_text = (
                            getattr(chunk, "text", "")[:200] if hasattr(chunk, "text") else ""
                        )
                        token_count = (
                            getattr(chunk, "token_count", 0)
                            if hasattr(chunk, "token_count")
                            else 0
                        )
                        typer.echo(f"\n--- Chunk {i} ({token_count} tokens) ---")
                        preview = chunk_text.encode("ascii", errors="replace").decode("ascii")
                        typer.echo(preview + ("..." if len(chunk_text) > 200 else ""))
                else:
                    typer.echo(f"Visualization error ({type(e).__name__}): {e}")
            except Exception as fallback_error:
                # If fallback also fails, show minimal error info without masking original
                typer.echo(
                    f"Visualization error ({type(e).__name__}) occurred, and fallback display also failed ({type(fallback_error).__name__})"
                )
                typer.echo(f"Original error: {e}")
                typer.echo(f"Fallback error: {fallback_error}")
    else:
        try:
            handshake_class = ComponentRegistry.get_handshake(handshaker).component_class
        except ValueError:
            typer.echo(
                f"Error: Unknown handshaker '{handshaker}'. Available: {', '.join(HANDSHAKES)}"
            )
            raise typer.Exit(code=1)

        typer.echo(f"Storing chunks in {handshaker}...")
        try:
            handshake_instance = handshake_class()
            handshake_instance.write(chunks)
            typer.echo("Chunks stored successfully.")
        except Exception as e:
            typer.echo(f"Error storing chunks: {e}")
            raise typer.Exit(code=1)


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
    chef_params: Optional[list[str]] = typer.Option(
        None,
        help="Parameters for the chef as key=value pairs (e.g., --chef-params clean_whitespace=true)",
    ),
    chunker: str = typer.Option(
        "semantic",
        help="Chunking method to use",
    ),
    chunk_size: Optional[int] = typer.Option(
        None,
        help="Maximum number of tokens per chunk",
    ),
    chunk_overlap: Optional[int] = typer.Option(
        None,
        help="Number of tokens to overlap between chunks",
    ),
    threshold: Optional[float] = typer.Option(
        None,
        help="Threshold for semantic similarity (0-1)",
    ),
    chunker_params: Optional[list[str]] = typer.Option(
        None,
        help="Additional parameters for the chunker as key=value pairs (e.g., --chunker-params tokenizer=gpt2 min_characters_per_chunk=50)",
    ),
    refiner: Optional[str] = typer.Option(
        None,
        help="Refiner method to use",
    ),
    refiner_params: Optional[list[str]] = typer.Option(
        None,
        help="Parameters for the refiner as key=value pairs (e.g., --refiner-params context_size=50)",
    ),
    handshaker: Optional[str] = typer.Option(
        None,
        help="Handshaker method to use",
    ),
    handshaker_params: Optional[list[str]] = typer.Option(
        None,
        help="Parameters for the handshaker as key=value pairs (e.g., --handshaker-params collection_name=documents)",
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
            chef_kwargs = parse_params(chef_params)
            pipe.process_with(chef, **chef_kwargs)

        # 3. Chunker
        explicit_chunker_params = {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "threshold": threshold,
        }
        parsed_chunker_params = parse_params(chunker_params)
        chunker_kwargs = merge_params(explicit_chunker_params, parsed_chunker_params)
        pipe.chunk_with(chunker, **chunker_kwargs)

        # 4. Refiner
        if refiner is not None:
            refiner_kwargs = parse_params(refiner_params)
            pipe.refine_with(refiner, **refiner_kwargs)

        # 5. Handshaker
        if handshaker is not None:
            handshaker_kwargs = parse_params(handshaker_params)
            pipe.store_in(handshaker, **handshaker_kwargs)

        # Run pipeline
        typer.echo("Running pipeline...")
        try:
            # If run_input is set, we pass it. If None, run() uses the fetcher step.
            doc = pipe.run(texts=run_input)
            # typer.echo(doc) # This prints the repr, which might be too verbose or ugly
        except Exception as e:
            typer.echo(f"Error running pipeline: {e}")
            raise typer.Exit(code=1)

        # Output results
        if handshaker:
            typer.echo(f"Pipeline completed and data stored in {handshaker}.")
            return

        if not doc:
            typer.echo("No output generated.")
            return

        docs: list[Document] = doc if isinstance(doc, list) else [doc]  # type: ignore

        # Set UTF-8 encoding for stdout on Windows if possible (once, before loop)
        if sys.platform == "win32":
            try:
                sys.stdout.reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass  # Python < 3.7 or reconfigure not available

        # Summary for multiple files
        if len(docs) > 1:
            total_chunks = sum(len(d.chunks) for d in docs)
            typer.echo(f"\nProcessed {len(docs)} files, {total_chunks} total chunks\n")

        for idx, d_obj in enumerate(docs, 1):
            # Print filename if available in metadata
            filename = (
                d_obj.metadata.get("filename", f"Document {idx}")
                if d_obj.metadata
                else f"Document {idx}"
            )
            if len(docs) > 1:
                typer.echo(f"\n{'=' * 80}")
                typer.echo(f"File {idx}/{len(docs)}: {filename} ({len(d_obj.chunks)} chunks)")
                typer.echo(f"{'=' * 80}\n")

            try:
                viz(d_obj.chunks)
            except (UnicodeEncodeError, UnicodeDecodeError, BrokenPipeError) as e:
                # Fallback for Windows console encoding issues
                try:
                    if d_obj.chunks:
                        typer.echo(f"Chunked into {len(d_obj.chunks)} chunks:")
                        for i, chunk in enumerate(d_obj.chunks, 1):
                            chunk_text = str(getattr(chunk, "text", ""))[:200]
                            token_count = getattr(chunk, "token_count", 0)
                            typer.echo(f"\n--- Chunk {i} ({token_count} tokens) ---")
                            # Truncate and escape problematic characters for display
                            preview = chunk_text.encode("ascii", errors="replace").decode("ascii")
                            typer.echo(preview + ("..." if len(chunk_text) > 200 else ""))
                    else:
                        typer.echo("No chunks to display (encoding error occurred)")
                except Exception as fallback_error:
                    # If fallback also fails, show minimal error info without masking original
                    typer.echo(
                        f"Encoding error ({type(e).__name__}) occurred, and fallback display also failed ({type(fallback_error).__name__})"
                    )
                    typer.echo(f"Original error: {e}")
                    typer.echo(f"Fallback error: {fallback_error}")
            except Exception as e:
                # Catch any other visualization errors and provide basic output
                try:
                    if d_obj.chunks:
                        typer.echo(
                            f"Chunked into {len(d_obj.chunks)} chunks (visualization error: {type(e).__name__})"
                        )
                        for i, chunk in enumerate(d_obj.chunks, 1):
                            chunk_text = (
                                getattr(chunk, "text", "")[:200] if hasattr(chunk, "text") else ""
                            )
                            token_count = (
                                getattr(chunk, "token_count", 0)
                                if hasattr(chunk, "token_count")
                                else 0
                            )
                            typer.echo(f"\n--- Chunk {i} ({token_count} tokens) ---")
                            preview = chunk_text.encode("ascii", errors="replace").decode("ascii")
                            typer.echo(preview + ("..." if len(chunk_text) > 200 else ""))
                    else:
                        typer.echo(f"Visualization error ({type(e).__name__}): {e}")
                except Exception as fallback_error:
                    # If fallback also fails, show minimal error info without masking original
                    typer.echo(
                        f"Visualization error ({type(e).__name__}) occurred, and fallback display also failed ({type(fallback_error).__name__})"
                    )
                    typer.echo(f"Original error: {e}")
                    typer.echo(f"Fallback error: {fallback_error}")

    except Exception as e:
        typer.echo(f"Pipeline error: {e}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
