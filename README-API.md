# Chonkie OSS API

A lightweight, self-hostable REST API that exposes the [Chonkie](https://github.com/chonkie-ai/chonkie) chunking library over HTTP.

No authentication, no billing, no cloud dependencies – just text in, chunks out.

---

## Quick Start

### Docker Compose (recommended)

```bash
# Clone the repo
git clone https://github.com/chonkie-ai/chonkie.git
cd chonkie

# Start the API (builds the image on first run)
docker compose up

# API is now available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### Docker Run

```bash
docker build -t chonkie-oss-api .
docker run -p 8000:8000 chonkie-oss-api
```

### Local Development

```bash
# Install Chonkie with API dependencies
pip install -e .[api,semantic,code,openai]

# Run the server
uvicorn api.main:app --reload --port 8000
```

---

## Configuration

All configuration is done via environment variables.

| Variable | Default | Description |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `CORS_ORIGINS` | `*` | Comma-separated list of allowed CORS origins |
| `OPENAI_API_KEY` | *(unset)* | Required only for `/v1/refine/embeddings` |

### Docker Compose with environment file

```bash
# .env
OPENAI_API_KEY=sk-...
LOG_LEVEL=DEBUG
CORS_ORIGINS=http://localhost:3000,https://myapp.example.com
```

```bash
docker compose --env-file .env up
```

---

## API Reference

Interactive docs are available at **`/docs`** (Swagger UI) and **`/redoc`** (ReDoc) when the server is running.

### Health Check

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

---

## Chunking Endpoints

### `POST /v1/chunk/token`

Splits text into fixed-size token windows.

**Request**

```json
{
  "text": "Your document text here...",
  "tokenizer": "character",
  "chunk_size": 512,
  "chunk_overlap": 0
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `string \| string[]` | *required* | Text or list of texts to chunk |
| `tokenizer` | `string` | `"character"` | Tokenizer (`"character"`, `"gpt2"`, `"cl100k_base"`, …) |
| `chunk_size` | `int` | `512` | Maximum tokens per chunk |
| `chunk_overlap` | `int` | `0` | Token overlap between adjacent chunks |

**Example**

```bash
curl -s -X POST http://localhost:8000/v1/chunk/token \
  -H "Content-Type: application/json" \
  -d '{"text": "The quick brown fox jumps over the lazy dog.", "chunk_size": 5}' \
  | python -m json.tool
```

---

### `POST /v1/chunk/sentence`

Splits text at sentence boundaries while respecting a token-size limit.

**Request**

```json
{
  "text": "First sentence. Second sentence. Third sentence.",
  "tokenizer": "character",
  "chunk_size": 512,
  "chunk_overlap": 0,
  "min_sentences_per_chunk": 1,
  "min_characters_per_sentence": 12,
  "approximate": false,
  "delim": ["\n", ". ", "! ", "? "],
  "include_delim": "prev"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `string \| string[]` | *required* | Text or list of texts |
| `tokenizer` | `string` | `"character"` | Tokenizer |
| `chunk_size` | `int` | `512` | Maximum tokens per chunk |
| `chunk_overlap` | `int` | `0` | Token overlap |
| `min_sentences_per_chunk` | `int` | `1` | Minimum sentences per chunk |
| `min_characters_per_sentence` | `int` | `12` | Minimum characters to count as a sentence |
| `approximate` | `bool` | `false` | Use approximate token counting for speed |
| `delim` | `string \| string[]` | `["\n",". ","! ","? "]` | Sentence delimiters |
| `include_delim` | `"prev" \| "next" \| null` | `"prev"` | Attach delimiter to previous or next sentence |

**Example**

```bash
curl -s -X POST http://localhost:8000/v1/chunk/sentence \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world. How are you? I am fine!", "chunk_size": 20}' \
  | python -m json.tool
```

---

### `POST /v1/chunk/recursive`

Recursively splits text using a hierarchy of structural separators defined by a named *recipe*.

**Request**

```json
{
  "text": "# Heading\n\nParagraph one.\n\nParagraph two.",
  "tokenizer": "character",
  "chunk_size": 512,
  "recipe": "markdown",
  "lang": "en",
  "min_characters_per_chunk": 24
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `string \| string[]` | *required* | Text or list of texts |
| `tokenizer` | `string` | `"character"` | Tokenizer |
| `chunk_size` | `int` | `512` | Maximum tokens per chunk |
| `recipe` | `string` | `"default"` | Splitting recipe (`"default"`, `"markdown"`, `"python"`, …) |
| `lang` | `string` | `"en"` | Language hint for the recipe |
| `min_characters_per_chunk` | `int` | `24` | Minimum characters per chunk |

**Example**

```bash
curl -s -X POST http://localhost:8000/v1/chunk/recursive \
  -H "Content-Type: application/json" \
  -d '{"text": "# Title\n\nSome paragraph text.\n\n## Section\n\nMore text.", "recipe": "markdown", "chunk_size": 100}' \
  | python -m json.tool
```

---

### `POST /v1/chunk/semantic`

Splits text based on semantic similarity between sentences using sentence embeddings.

> **Note:** Requires `chonkie[semantic]` to be installed.  The default model (`minishlab/potion-base-8M`) is downloaded automatically on first use.

**Request**

```json
{
  "text": "The sky is blue. Water is wet. Apples are red.",
  "embedding_model": "minishlab/potion-base-8M",
  "threshold": 0.5,
  "chunk_size": 512
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `string \| string[]` | *required* | Text or list of texts |
| `embedding_model` | `string` | `"minishlab/potion-base-8M"` | Sentence embedding model |
| `threshold` | `float` | `0.5` | Cosine-similarity split threshold (0–1) |
| `chunk_size` | `int` | `512` | Maximum tokens per chunk |
| `similarity_window` | `int` | `3` | Surrounding sentences used for similarity |
| `min_sentences_per_chunk` | `int` | `1` | Minimum sentences per chunk |
| `min_characters_per_sentence` | `int` | `12` | Minimum sentence length |
| `delim` | `string \| string[]` | `["\n",". ","! ","? "]` | Sentence delimiters |
| `include_delim` | `"prev" \| "next" \| null` | `"prev"` | Delimiter attachment |
| `skip_window` | `int` | `0` | Skip window for similarity computation |
| `filter_window` | `int` | `5` | Savitzky-Golay filter window |
| `filter_polyorder` | `int` | `3` | Savitzky-Golay polynomial order |
| `filter_tolerance` | `float` | `0.2` | Peak detection tolerance |

**Example**

```bash
curl -s -X POST http://localhost:8000/v1/chunk/semantic \
  -H "Content-Type: application/json" \
  -d '{"text": "Dogs are great pets. Cats are independent animals. The stock market rose today.", "threshold": 0.4}' \
  | python -m json.tool
```

---

### `POST /v1/chunk/code`

Splits source code at AST-level boundaries (functions, classes, etc.) using tree-sitter.

> **Note:** Requires `chonkie[code]` to be installed.

**Request**

```json
{
  "text": "def foo():\n    pass\n\ndef bar():\n    pass\n",
  "tokenizer": "character",
  "chunk_size": 512,
  "language": "python",
  "include_nodes": false
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `text` | `string \| string[]` | *required* | Source code or list of source code snippets |
| `tokenizer` | `string` | `"character"` | Tokenizer |
| `chunk_size` | `int` | `512` | Maximum tokens per chunk |
| `language` | `string` | `"python"` | Programming language (`"python"`, `"javascript"`, `"java"`, …) |
| `include_nodes` | `bool` | `false` | Include AST node metadata in chunk output |

**Example**

```bash
curl -s -X POST http://localhost:8000/v1/chunk/code \
  -H "Content-Type: application/json" \
  -d '{"text": "def hello():\n    print(\"hello\")\n\ndef world():\n    print(\"world\")\n", "language": "python", "chunk_size": 50}' \
  | python -m json.tool
```

---

## Refinery Endpoints

Refineries post-process chunks returned by a chunking endpoint.  They accept
a list of chunk dicts (as returned by any `/v1/chunk/*` endpoint) and return an
enriched list.

### `POST /v1/refine/embeddings`

Attaches OpenAI embedding vectors to each chunk.

> **Note:** Requires `OPENAI_API_KEY` environment variable and `chonkie[openai]`.

**Supported models:** `text-embedding-3-small`, `text-embedding-3-large`

**Request**

```json
{
  "chunks": [
    {"text": "Hello world", "start_index": 0, "end_index": 11, "token_count": 2}
  ],
  "embedding_model": "text-embedding-3-small"
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `chunks` | `ChunkDict[]` | *required* | Chunks to embed (from a chunking endpoint) |
| `embedding_model` | `string` | `"text-embedding-3-small"` | OpenAI embedding model |

**Example (two-step: chunk then embed)**

```bash
# Step 1: chunk
CHUNKS=$(curl -s -X POST http://localhost:8000/v1/chunk/token \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world. How are you?", "chunk_size": 10}')

# Step 2: embed
curl -s -X POST http://localhost:8000/v1/refine/embeddings \
  -H "Content-Type: application/json" \
  -d "{\"chunks\": $CHUNKS, \"embedding_model\": \"text-embedding-3-small\"}" \
  | python -m json.tool
```

---

### `POST /v1/refine/overlap`

Appends or prepends overlapping context from neighbouring chunks, improving
continuity for RAG and search use-cases.

**Request**

```json
{
  "chunks": [
    {"text": "First chunk.", "start_index": 0, "end_index": 12, "token_count": 2},
    {"text": "Second chunk.", "start_index": 13, "end_index": 26, "token_count": 2}
  ],
  "tokenizer": "character",
  "context_size": 0.25,
  "mode": "token",
  "method": "suffix",
  "merge": true
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `chunks` | `ChunkDict[]` | *required* | Chunks to refine |
| `tokenizer` | `string` | `"character"` | Tokenizer |
| `context_size` | `float \| int` | `0.25` | Overlap size (fraction of chunk or absolute token count) |
| `mode` | `"token" \| "recursive"` | `"token"` | Strategy for creating the overlap window |
| `method` | `"suffix" \| "prefix"` | `"suffix"` | Append context from previous (`suffix`) or next (`prefix`) chunk |
| `merge` | `bool` | `true` | Merge context into chunk text |

**Example**

```bash
curl -s -X POST http://localhost:8000/v1/refine/overlap \
  -H "Content-Type: application/json" \
  -d '{
    "chunks": [
      {"text": "The quick brown fox", "start_index": 0, "end_index": 19, "token_count": 19},
      {"text": "jumps over the lazy dog.", "start_index": 20, "end_index": 44, "token_count": 24}
    ],
    "context_size": 5
  }' \
  | python -m json.tool
```

---

## Response Format

All chunking endpoints return a JSON array of chunk objects:

```json
[
  {
    "text": "chunk text...",
    "start_index": 0,
    "end_index": 100,
    "token_count": 50
  },
  ...
]
```

When the input is a **list of texts**, the response is a **list of lists**:

```json
[
  [{"text": "...", "start_index": 0, "end_index": 50, "token_count": 10}],
  [{"text": "...", "start_index": 0, "end_index": 75, "token_count": 15}]
]
```

After using `/v1/refine/embeddings`, each chunk also includes an `embedding` field:

```json
{
  "text": "chunk text...",
  "start_index": 0,
  "end_index": 100,
  "token_count": 50,
  "embedding": [0.012, -0.034, ...]
}
```

---

## Deployment Notes

### Production

For production deployments it is recommended to:

1. Pin the Docker image tag (e.g. `chonkie-oss-api:1.5.6`)
2. Set `CORS_ORIGINS` to your frontend domain(s)
3. Run behind a reverse proxy (nginx, Caddy) for TLS termination
4. Set resource limits in your container orchestrator (CPU/memory)

### Uvicorn workers

To handle concurrent requests more efficiently, increase the number of workers:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Or override the Docker CMD:

```yaml
# docker-compose.yml
command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Model caching

The `SemanticChunker` downloads its embedding model from the Hugging Face Hub on
first use.  In containerised deployments you can pre-bake the model into the
image or mount a persistent volume at `~/.cache/huggingface`.

```yaml
volumes:
  - hf_cache:/root/.cache/huggingface
```

---

## License

Apache 2.0 – see [LICENSE](LICENSE).
