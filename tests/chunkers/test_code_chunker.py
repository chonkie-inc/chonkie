"""Test the CodeChunker class."""
import pytest

from chonkie.chunker import CodeChunker, UnsupportedLanguageException
from chonkie.types.code import CodeChunk


@pytest.fixture
def python_code() -> str:
    """Return a sample Python code snippet."""
    return """
import os
import sys

def hello_world(name: str):
    \"\"\"Prints a greeting.\"\"\"
    print(f"Hello, {name}!")

class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value

if __name__ == "__main__":
    hello_world("World")
    instance = MyClass(10)
    print(instance.get_value())
"""


@pytest.fixture
def js_code() -> str:
    """Return a sample JavaScript code snippet."""
    return """
function greet(name) {
  console.log(`Hello, ${name}!`);
}

class Calculator {
  add(a, b) {
    return a + b;
  }
}

const calc = new Calculator();
greet('Developer');
console.log(calc.add(5, 3));
"""


@pytest.fixture
def unsupported_code() -> str:
    """Return a sample ASP.NET code snippet.
    
    ASP.NET is one of the languages that is not supported by the tree-sitter-language-pack yet.
    But, it can be detected by Magika.
    """
    return """
<%@ Page Language="C#" %>
<!DOCTYPE html>
<html>
<head>
    <title>ASP.NET Test</title>
</head>
<body>
    <h1>ASP.NET Inline Example</h1>
    <%
        string message = "Hello from ASP.NET!";
        Response.Write(message);
    %>
</body>
</html>
"""


@pytest.fixture
def unsupported_language() -> str:
    """Return a language unsupported by the tree-sitter-language-pack."""
    # asp.net is one of them
    return "asp"


@pytest.fixture
def plain_text() -> str:
    """Return a sample plain text."""
    text = """# Chunking Strategies in Retrieval-Augmented Generation: A Comprehensive Analysis\n\nIn the rapidly evolving landscape of natural language processing, Retrieval-Augmented Generation (RAG) has emerged as a groundbreaking approach that bridges the gap between large language models and external knowledge bases. At the heart of these systems lies a crucial yet often overlooked process: chunking. This fundamental operation, which involves the systematic decomposition of large text documents into smaller, semantically meaningful units, plays a pivotal role in determining the overall effectiveness of RAG implementations.\n\nThe process of text chunking in RAG applications represents a delicate balance between competing requirements. On one side, we have the need for semantic coherence – ensuring that each chunk maintains meaningful context that can be understood and processed independently. On the other, we must optimize for information density, ensuring that each chunk carries sufficient signal without excessive noise that might impede retrieval accuracy. This balancing act becomes particularly crucial when we consider the downstream implications for vector databases and embedding models that form the backbone of modern RAG systems.\n\nThe selection of appropriate chunk size emerges as a fundamental consideration that significantly impacts system performance. Through extensive experimentation and real-world implementations, researchers have identified that chunks typically perform optimally in the range of 256 to 1024 tokens. However, this range should not be treated as a rigid constraint but rather as a starting point for optimization based on specific use cases and requirements. The implications of chunk size selection ripple throughout the entire RAG pipeline, affecting everything from storage requirements to retrieval accuracy and computational overhead.\n\nFixed-size chunking represents the most straightforward approach to document segmentation, offering predictable memory usage and consistent processing time. However, this apparent simplicity comes with significant drawbacks. By arbitrarily dividing text based on token or character count, fixed-size chunking risks fragmenting semantic units and disrupting the natural flow of information. Consider, for instance, a technical document where a complex concept is explained across several paragraphs – fixed-size chunking might split this explanation at critical junctures, potentially compromising the system's ability to retrieve and present this information coherently.\n\nIn response to these limitations, semantic chunking has gained prominence as a more sophisticated alternative. This approach leverages natural language understanding to identify meaningful boundaries within the text, respecting the natural structure of the document. Semantic chunking can operate at various levels of granularity, from simple sentence-based segmentation to more complex paragraph-level or topic-based approaches. The key advantage lies in its ability to preserve the inherent semantic relationships within the text, leading to more meaningful and contextually relevant retrieval results.\n\nRecent advances in the field have given rise to hybrid approaches that attempt to combine the best aspects of both fixed-size and semantic chunking. These methods typically begin with semantic segmentation but impose size constraints to prevent extreme variations in chunk length. Furthermore, the introduction of sliding window techniques with overlap has proved particularly effective in maintaining context across chunk boundaries. This overlap, typically ranging from 10% to 20% of the chunk size, helps ensure that no critical information is lost at segment boundaries, albeit at the cost of increased storage requirements.\n\nThe implementation of chunking strategies must also consider various technical factors that can significantly impact system performance. Vector database capabilities, embedding model constraints, and runtime performance requirements all play crucial roles in determining the optimal chunking approach. Moreover, content-specific factors such as document structure, language characteristics, and domain-specific requirements must be carefully considered. For instance, technical documentation might benefit from larger chunks that preserve detailed explanations, while news articles might perform better with smaller, more focused segments.\n\nThe future of chunking in RAG systems points toward increasingly sophisticated approaches. Current research explores the potential of neural chunking models that can learn optimal segmentation strategies from large-scale datasets. These models show promise in adapting to different content types and query patterns, potentially leading to more efficient and effective retrieval systems. Additionally, the emergence of cross-lingual chunking strategies addresses the growing need for multilingual RAG applications, while real-time adaptive chunking systems attempt to optimize segment boundaries based on user interaction patterns and retrieval performance metrics.\n\nThe effectiveness of RAG systems heavily depends on the thoughtful implementation of appropriate chunking strategies. While the field continues to evolve, practitioners must carefully consider their specific use cases and requirements when designing chunking solutions. Factors such as document characteristics, retrieval patterns, and performance requirements should guide the selection and optimization of chunking strategies. As we look to the future, the continued development of more sophisticated chunking approaches promises to further enhance the capabilities of RAG systems, enabling more accurate and efficient information retrieval and generation.\n\nThrough careful consideration of these various aspects and continued experimentation with different approaches, organizations can develop chunking strategies that effectively balance the competing demands of semantic coherence, computational efficiency, and retrieval accuracy. As the field continues to evolve, we can expect to see new innovations that further refine our ability to segment and process textual information in ways that enhance the capabilities of RAG systems while maintaining their practical utility in real-world applications."""
    return text


def test_code_chunker_initialization() -> None:
    """Test CodeChunker initialization."""
    chunker = CodeChunker(language="python", chunk_size=128)
    assert chunker.chunk_size == 128
    assert chunker.return_type == "chunks"
    assert chunker.parser is not None


def test_code_chunker_chunking_python(python_code: str) -> None:
    """Test basic chunking of Python code."""
    chunker = CodeChunker(language="python", chunk_size=50, include_nodes=True)
    chunks = chunker.chunk(python_code)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, CodeChunk) for chunk in chunks)
    assert all(chunk.text is not None for chunk in chunks)
    assert all(chunk.start_index is not None for chunk in chunks)
    assert all(chunk.end_index is not None for chunk in chunks)
    assert all(chunk.token_count is not None for chunk in chunks)
    assert all(chunk.nodes is not None for chunk in chunks)


def test_code_chunker_reconstruction_python(python_code: str) -> None:
    """Test if the original Python code can be reconstructed from chunks."""
    chunker = CodeChunker(language="python", chunk_size=50)
    chunks = chunker.chunk(python_code)
    reconstructed_text = "".join(chunk.text for chunk in chunks)
    assert reconstructed_text == python_code


def test_code_chunker_chunk_size_python(python_code: str) -> None:
    """Test if Python code chunks mostly adhere to chunk_size."""
    chunk_size = 50
    chunker = CodeChunker(language="python", chunk_size=chunk_size)
    chunks = chunker.chunk(python_code)
    # Allow for some leeway as splitting happens at node boundaries
    assert all(chunk.token_count < chunk_size + 20 for chunk in chunks[:-1]) # Check all but last chunk rigorously
    assert chunks[-1].token_count > 0 # Last chunk must have content


def test_code_chunker_indices_python(python_code: str) -> None:
    """Test the start and end indices of Python code chunks."""
    chunker = CodeChunker(language="python", chunk_size=50)
    chunks = chunker.chunk(python_code)
    current_index = 0
    for chunk in chunks:
        assert chunk.start_index == current_index
        assert chunk.end_index == current_index + len(chunk.text)
        assert chunk.text == python_code[chunk.start_index:chunk.end_index]
        current_index = chunk.end_index
    assert current_index == len(python_code)


def test_code_chunker_return_type_texts(python_code: str) -> None:
    """Test return_type='texts'."""
    chunker = CodeChunker(language="python", chunk_size=50, return_type="texts")
    texts = chunker.chunk(python_code)
    assert isinstance(texts, list)
    assert len(texts) > 0
    assert all(isinstance(text, str) for text in texts)
    reconstructed_text = "".join(texts)
    assert reconstructed_text == python_code


def test_code_chunker_empty_input() -> None:
    """Test chunking an empty string."""
    chunker = CodeChunker(language="python")
    chunks = chunker.chunk("")
    assert chunks == []

    # Test return_type='texts'
    chunker = CodeChunker(language="python", return_type="texts")
    texts = chunker.chunk("")
    assert texts == []


def test_code_chunker_whitespace_input() -> None:
    """Test chunking a string with only whitespace."""
    chunker = CodeChunker(language="python")
    chunks = chunker.chunk("   \n\t\n  ")
    assert chunks == []

    # Test return_type='texts'
    chunker = CodeChunker(language="python", return_type="texts")
    texts = chunker.chunk("   \n\t\n  ")
    assert texts == []


def test_code_chunker_chunking_javascript(js_code: str) -> None:
    """Test basic chunking of JavaScript code."""
    chunker = CodeChunker(language="javascript", chunk_size=30)
    chunks = chunker.chunk(js_code)

    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert all(isinstance(chunk, CodeChunk) for chunk in chunks)


def test_code_chunker_reconstruction_javascript(js_code: str) -> None:
    """Test if the original JavaScript code can be reconstructed."""
    chunker = CodeChunker(language="javascript", chunk_size=30)
    chunks = chunker.chunk(js_code)
    reconstructed_text = "".join(chunk.text for chunk in chunks)
    assert reconstructed_text == js_code


def test_code_chunker_chunk_size_javascript(js_code: str) -> None:
    """Test if JavaScript code chunks mostly adhere to chunk_size."""
    chunk_size = 30
    chunker = CodeChunker(language="javascript", chunk_size=chunk_size)
    chunks = chunker.chunk(js_code)
    # Allow for some leeway
    assert all(chunk.token_count < chunk_size + 15 for chunk in chunks[:-1])
    assert chunks[-1].token_count > 0 
    
    
def test_code_chunker_chunking_unsupported_input(unsupported_code: str) -> None:
    """Test if the UnsupportedLanguageException is being raised if an unsupported input is given."""
    chunk_size = 30
    # language set to "auto", letting the chonkie detect the language 
    chunker = CodeChunker(language="auto", chunk_size=chunk_size)
    with pytest.raises(UnsupportedLanguageException) as exc_info:
        chunker.chunk(unsupported_code)
    
    assert isinstance(exc_info.value, UnsupportedLanguageException)
    
    
def test_code_chunker_chunking_unsupported_language(unsupported_language: str) -> None:
    """Test if the UnsupportedLanguageException is being raised if an unsupported language is given."""
    chunk_size = 30
    with pytest.raises(UnsupportedLanguageException) as exc_info:
        CodeChunker(language=unsupported_language, chunk_size=chunk_size)
        
    assert isinstance(exc_info.value, UnsupportedLanguageException)
    
    
def test_code_chunker_chunking_plain_text(plain_text: str) -> None:
    """Test if the CodeChunker is falling back to the RecursiveChunker if the input is detected as 'txt' or the language given is 'txt'."""
    chunk_size = 30
    with pytest.raises(UnsupportedLanguageException) as exc_info:
        CodeChunker(language="txt", chunk_size=chunk_size)
        
    assert isinstance(exc_info.value, UnsupportedLanguageException)
    
    chunker = CodeChunker(language="auto", chunk_size=chunk_size)
    with pytest.raises(UnsupportedLanguageException) as exc_info:
        chunker.chunk(plain_text)

    assert isinstance(exc_info.value, UnsupportedLanguageException)