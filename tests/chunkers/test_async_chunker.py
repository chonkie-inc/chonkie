"""Tests for the async chunker functionality."""

import asyncio
from typing import List, Sequence, cast

import pytest
import tiktoken
from tiktoken import Encoding

from chonkie import Chunk, TokenChunker
from chonkie.chunker.base import BaseChunker, T


@pytest.fixture
def tiktokenizer() -> Encoding:
    """Fixture that returns a GPT-2 tokenizer from the tiktoken library."""
    return tiktoken.get_encoding("gpt2")


@pytest.fixture
def sample_text() -> str:
    """Fixture that returns a sample text for testing."""
    text = """According to all known laws of aviation, there is no way a bee should be able to fly. Its wings are too small to get its fat little body off the ground. The bee, of course, flies anyway because bees don't care what humans think is impossible. Yellow, black. Yellow, black. Yellow, black. Yellow, black. Ooh, black and yellow! Let's shake it up a little. Barry! Breakfast is ready! Coming! Hang on a second. Hello? - Barry? - Adam? - Can you believe this is happening? - I can't. I'll pick you up. Looking sharp. Use the stairs. Your father paid good money for those. Sorry. I'm excited. Here's the graduate. We're very proud of you, son. A perfect report card, all B's. Very proud. Ma! I got a thing going here."""
    return text


@pytest.fixture
def sample_batch(sample_text: str) -> List[str]:
    """Fixture that returns a sample batch of texts for testing."""
    batch = []
    base_text = sample_text + " "  # Add space for separation when repeating
    
    # Create 10 texts with varying lengths
    for i in range(10):
        repeats = 1 + (i % 3)  # Cycle through 1, 2, 3 repeats
        batch.append(base_text * repeats)
        
    return batch


@pytest.mark.asyncio
async def test_async_chunk_single_text(tiktokenizer: Encoding, sample_text: str) -> None:
    """Test that a chunker can asynchronously chunk a single text."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = await chunker.async_chunk(sample_text)

    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)


@pytest.mark.asyncio
async def test_stream_chunk(tiktokenizer: Encoding, sample_text: str) -> None:
    """Test that a chunker can stream chunks as they're generated."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = []
    
    async for chunk in chunker.stream_chunk(sample_text):
        chunks.append(chunk)
    
    assert len(chunks) > 0
    assert isinstance(chunks[0], Chunk)
    assert all(chunk.token_count <= 512 for chunk in chunks)
    assert all(chunk.token_count > 0 for chunk in chunks)


@pytest.mark.asyncio
async def test_async_chunk_batch(tiktokenizer: Encoding, sample_batch: List[str]) -> None:
    """Test that a chunker can asynchronously chunk a batch of texts."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunk_batches = await chunker.async_chunk_batch(sample_batch, show_progress=False)
    
    assert len(chunk_batches) == len(sample_batch)
    for chunks in chunk_batches:
        assert len(chunks) > 0
        assert isinstance(chunks[0], Chunk)
        assert all(chunk.token_count <= 512 for chunk in chunks)
        assert all(chunk.token_count > 0 for chunk in chunks)


@pytest.mark.asyncio
async def test_async_chunk_empty_text(tiktokenizer: Encoding) -> None:
    """Test that a chunker can asynchronously handle an empty text."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunks = await chunker.async_chunk("")
    
    assert len(chunks) == 0


@pytest.mark.asyncio
async def test_async_chunk_batch_empty(tiktokenizer: Encoding) -> None:
    """Test that a chunker can asynchronously handle an empty batch."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunk_batches = await chunker.async_chunk_batch([])
    
    assert len(chunk_batches) == 0


@pytest.mark.asyncio
async def test_async_chunk_batch_single(tiktokenizer: Encoding, sample_text: str) -> None:
    """Test that a chunker can asynchronously handle a batch with a single text."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    chunk_batches = await chunker.async_chunk_batch([sample_text])
    
    assert len(chunk_batches) == 1
    assert len(chunk_batches[0]) > 0


@pytest.mark.asyncio
async def test_configuration(tiktokenizer: Encoding, sample_batch: List[str]) -> None:
    """Test the configuration of chunker processing options."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    
    # Configure for lower concurrency
    chunker.configure(max_concurrency=3, batch_size=2)
    assert chunker._max_concurrency == 3
    assert chunker._batch_size == 2
    
    # Test with new configuration
    chunk_batches = await chunker.async_chunk_batch(sample_batch[:4], show_progress=False)
    assert len(chunk_batches) == 4


@pytest.mark.asyncio
async def test_concurrency_limiting(tiktokenizer: Encoding, sample_batch: List[str]) -> None:
    """Test that concurrency is properly limited."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    
    # Set very small concurrency and batch size to test limiting
    chunker.configure(max_concurrency=2, batch_size=3)
    
    # Track concurrent executions
    active_count = 0
    max_active = 0
    lock = asyncio.Lock()
    
    original_async_chunk = chunker.async_chunk
    
    async def tracked_async_chunk(text: str) -> Sequence[T]:
        nonlocal active_count, max_active
        async with lock:
            active_count += 1
            max_active = max(max_active, active_count)
        
        result = await original_async_chunk(text)
        
        async with lock:
            active_count -= 1
        
        return result
    
    # Monkey patch the async_chunk method to track concurrency
    chunker.async_chunk = tracked_async_chunk  # type: ignore
    
    # Process a batch with our tracked method
    await chunker.async_chunk_batch(sample_batch[:6], show_progress=False)
    
    # Check that max concurrency was not exceeded
    assert max_active <= chunker._max_concurrency


@pytest.mark.asyncio
async def test_multiprocessing_config(tiktokenizer: Encoding) -> None:
    """Test that multiprocessing configuration works."""
    chunker = TokenChunker(tokenizer=tiktokenizer, chunk_size=512, chunk_overlap=128)
    
    # Test initial state - TokenChunker defaults to False for multiprocessing
    assert chunker._use_multiprocessing is False
    
    # Test changing multiprocessing setting
    chunker.configure(use_multiprocessing=True)
    assert chunker._use_multiprocessing is True
    
    # Test method chaining
    result = chunker.configure(use_multiprocessing=False)
    assert result is chunker  # Should return self for method chaining
    assert chunker._use_multiprocessing is False 