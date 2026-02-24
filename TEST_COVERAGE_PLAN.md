# Test Coverage Plan for Chonkie

## Current State

The Chonkie test suite has **62 test files** covering the major chunking and embedding pipelines, but leaves several significant modules untested or partially tested.

Running a rough estimate, the core chunkers and pipeline have solid coverage, but the following areas are significantly under-covered:

| Module | Source Files | Test Files | Coverage Estimate |
|--------|-------------|------------|-------------------|
| `types/` | 7 | 3 | ~40% |
| `experimental/` | 3 | 0 | ~0% |
| `cloud/refineries/` | 4 | 0 | ~0% |
| `handshakes/` (turbopuffer + utils) | 2 | 0 | ~0% |
| `genie/` (cerebras + groq) | 2 | 0 | ~0% |
| `utils/` (_api.py, hub.py) | 4 | 1 (partial) | ~20% |
| `pipeline/` (export/store/config) | 4 | 1 (partial) | ~55% |
| `porters/` (base) | 4 | 2 (no base) | ~55% |

---

## Gaps to Address

### Priority 1 — Types Module (`src/chonkie/types/`)

**Missing test files:** `test_document.py`, `test_code.py`, `test_markdown.py`

The `sentence.py` type has rich validation logic (`__post_init__`, `to_dict`, `from_dict`) but its tests are minimal. The `document.py`, `code.py`, and `markdown.py` types have no dedicated tests at all.

#### `tests/types/test_document.py` (new file)

```
test_document_initialization
  - Default factory creates empty chunks list and metadata dict
  - ID is auto-generated with "doc" prefix
  - content field stores string

test_document_with_chunks
  - Chunk list accepts standard Chunk objects
  - Chunks list is mutable (append/extend)

test_document_metadata
  - Metadata dict accepts arbitrary key-value pairs
  - Metadata persists between accesses

test_document_id_uniqueness
  - Two Document instances get different auto-generated IDs
```

#### `tests/types/test_code.py` (new file)

```
test_merge_rule_basic
  - MergeRule with required fields
  - Default bidirectional=False

test_merge_rule_with_pattern
  - Optional text_pattern accepts regex string

test_split_rule_string_body_child
  - body_child as a simple string path

test_split_rule_list_body_child
  - body_child as a list of strings (multi-level path)

test_split_rule_exclude_nodes
  - Optional exclude_nodes accepts list of node type strings

test_language_config_composition
  - LanguageConfig holds both merge_rules and split_rules lists
  - Empty lists as defaults
```

#### `tests/types/test_markdown.py` (new file)

```
test_markdown_table
  - Content and index attributes stored correctly

test_markdown_code_with_language
  - Optional language field set and retrieved

test_markdown_code_without_language
  - language defaults to None

test_markdown_image_with_link
  - Optional link field set and retrieved

test_markdown_image_without_link
  - link defaults to None

test_markdown_document_inherits_document
  - MarkdownDocument is a subclass of Document
  - tables, code, images default to empty lists

test_markdown_document_fields
  - tables/code/images lists are mutable
```

#### `tests/types/test_sentence.py` (extend existing)

```
test_sentence_post_init_invalid_text_type
  - Raises TypeError if text is not a string

test_sentence_post_init_negative_start_index
  - Raises ValueError for start_index < 0

test_sentence_post_init_invalid_index_order
  - Raises ValueError if start_index > end_index

test_sentence_post_init_negative_token_count
  - Raises ValueError for token_count < 0

test_sentence_repr_format
  - __repr__ includes text, start/end indices, token_count

test_sentence_to_dict_with_numpy
  - numpy array embedding is serialized to list

test_sentence_from_dict_round_trip
  - to_dict then from_dict returns equivalent object
```

---

### Priority 2 — Utility Functions (`src/chonkie/utils/`)

**Affected files:** `_api.py`, `hub.py`

These handle API key management (login, load_token) and Hubbie (get_pipeline_recipe). Since they touch the filesystem and network, they must be tested with mocking.

#### `tests/test_utils.py` (extend existing)

```
# --- _api.py ---

test_get_config_path_returns_string
  - get_config_path() returns a path ending in config.json

test_get_config_path_creates_directory
  - tmp_path monkeypatch: calling get_config_path() creates ~/.chonkie/ dir

test_login_writes_api_key
  - login("test-key") writes key to config file
  - Config JSON contains {"api_key": "test-key"}

test_login_updates_existing_config
  - login() with existing config replaces old key

test_login_handles_malformed_json
  - login() overwrites malformed existing config gracefully

test_load_token_from_env_variable
  - CHONKIE_API_KEY env var is returned without reading file

test_load_token_from_config_file
  - Token read from config file when env var absent

test_load_token_env_takes_priority_over_file
  - If both env var and config file exist, env var wins

test_load_token_raises_if_no_source
  - ValueError raised if env var missing and config file absent

test_load_token_raises_if_api_key_not_in_config
  - ValueError raised if config file exists but has no api_key

# --- hub.py ---

test_hubbie_get_pipeline_recipe
  - get_pipeline_recipe() returns a valid pipeline configuration dict

test_hubbie_invalid_recipe_raises
  - Requesting a non-existent recipe raises an appropriate error
```

---

### Priority 3 — Handshakes Utils (`src/chonkie/handshakes/utils.py`)

**Missing test file:** `tests/handshakes/test_handshake_utils.py`

```
test_generate_random_collection_name_format
  - Output has three parts separated by "-"
  - Each part is a non-empty string

test_generate_random_collection_name_custom_separator
  - generate_random_collection_name(sep="_") uses underscore separator

test_generate_random_collection_name_randomness
  - Two calls typically return different names (not deterministic)

test_generate_random_collection_name_word_pool
  - First word comes from ADJECTIVES, second from VERBS, third from NOUNS
  - All words are lowercase alphabetic strings
```

---

### Priority 4 — Cloud Refineries (`src/chonkie/cloud/refineries/`)

**Missing test file:** `tests/cloud/test_cloud_refineries.py`

These make HTTP calls to `api.chonkie.ai`. All network calls must be mocked with `unittest.mock.patch`.

#### BaseRefinery

```
test_base_refinery_is_abstract
  - Instantiating BaseRefinery directly raises TypeError

test_base_refinery_call_delegates_to_refine
  - __call__ invokes refine() with same arguments
```

#### OverlapRefinery

```
test_overlap_refinery_raises_without_api_key
  - ValueError raised when api_key is None and CHONKIE_API_KEY not set

test_overlap_refinery_loads_api_key_from_env
  - CHONKIE_API_KEY env var is used when api_key=None

test_overlap_refinery_stores_all_parameters
  - All constructor args (tokenizer, context_size, mode, method, etc.) stored

test_overlap_refinery_refine_sends_correct_payload (mocked)
  - POST body contains chunk serializations (to_dict)
  - Authorization header contains api_key
  - URL is {BASE_URL}/{VERSION}/refine/overlap

test_overlap_refinery_refine_deserializes_response (mocked)
  - from_dict called on each item in response JSON

test_overlap_refinery_raises_on_mixed_chunk_types
  - ValueError raised when chunks list contains different Chunk subclasses

test_overlap_refinery_token_mode
  - context_size as float (0.0–1.0) accepted in "token" mode

test_overlap_refinery_recursive_mode
  - context_size as int accepted in "recursive" mode
```

#### EmbeddingsRefinery

```
test_embeddings_refinery_raises_without_api_key
  - ValueError raised when api_key is None and env var not set

test_embeddings_refinery_loads_api_key_from_env
  - CHONKIE_API_KEY env var is used when api_key=None

test_embeddings_refinery_refine_sends_correct_payload (mocked)
  - POST body contains serialized chunks
  - URL is {BASE_URL}/{VERSION}/refine/embeddings
  - Authorization header present

test_embeddings_refinery_attaches_embeddings (mocked)
  - Response embeddings converted to numpy arrays
  - Embeddings attached to each returned chunk

test_embeddings_refinery_raises_on_mixed_chunk_types
  - ValueError raised for heterogeneous chunk types
```

---

### Priority 5 — Experimental CodeChunker (`src/chonkie/experimental/`)

**Missing test file:** `tests/experimental/test_code_chunker.py`

This is the most complex untested module (~650 lines of AST-based chunking logic). Tests should cover the most important language paths without requiring all 13+ language grammars to be installed.

```
# Fixtures
@pytest.fixture
def python_code() -> str:
    """Short Python snippet with a class and two methods."""

@pytest.fixture
def javascript_code() -> str:
    """Short JS snippet with a function and an arrow function."""

# Basic initialization
test_code_chunker_init_default
  - Instantiates without arguments

test_code_chunker_init_with_language
  - Instantiates with explicit language="python"

test_code_chunker_not_available_without_deps
  - _is_available() returns False if tree-sitter not installed
  - (skip test if deps are present)

# Chunking correctness
test_code_chunker_python_basic
  - chunks() returns non-empty list of CodeChunk objects
  - All chunk texts are substrings of the original code
  - Chunk indices are valid (text[start:end] == chunk.text)

test_code_chunker_python_function_split
  - Top-level functions end up in separate chunks when they exceed chunk_size

test_code_chunker_python_class_split
  - Large class methods are split into separate chunks

test_code_chunker_javascript_basic
  - chunks() works on JavaScript input

test_code_chunker_empty_text
  - Returns empty list for empty string input

test_code_chunker_single_small_function
  - Code smaller than chunk_size returns a single chunk

test_code_chunker_batch_chunking
  - chunk_batch() returns one list of chunks per input text

test_code_chunker_language_detection
  - _detect_language() returns a known language name for Python code
  - (skip if magika not installed)

test_code_chunker_token_counts
  - Each chunk's token_count > 0
  - Sum of chunk token_counts is reasonable relative to full text length

# CodeLanguageRegistry
test_code_registry_has_python_config
  - Registry provides a LanguageConfig for "python"

test_code_registry_merge_rules_non_empty
  - Python config has at least one MergeRule

test_code_registry_split_rules_non_empty
  - Python config has at least one SplitRule

test_code_registry_unknown_language
  - Returns None (or raises) for completely unknown language tag
```

---

### Priority 6 — Genie Providers (`src/chonkie/genie/`)

**Missing test files:** `tests/genie/test_cerebras_genie.py`, `tests/genie/test_groq_genie.py`

These require API keys; all network calls must be mocked.

#### Cerebras Genie

```
test_cerebras_genie_init_without_key_raises
  - ValueError when no CEREBRAS_API_KEY and api_key=None

test_cerebras_genie_init_with_env_key
  - Reads key from CEREBRAS_API_KEY env var

test_cerebras_genie_complete_mocked
  - _complete() sends correct messages payload
  - Returns string response from mock

test_cerebras_genie_not_available_without_dep
  - _is_available() returns False when cerebras SDK not installed
```

#### Groq Genie

```
test_groq_genie_init_without_key_raises
  - ValueError when no GROQ_API_KEY and api_key=None

test_groq_genie_init_with_env_key
  - Reads key from GROQ_API_KEY env var

test_groq_genie_complete_mocked
  - _complete() sends correct messages payload
  - Returns string response from mock

test_groq_genie_not_available_without_dep
  - _is_available() returns False when groq SDK not installed
```

---

### Priority 7 — Pipeline Gaps (`tests/test_pipeline.py`)

The existing pipeline tests are thorough for chunking workflows but miss several methods.

```
# export_with (Porter step)
test_pipeline_export_with_json_porter
  - Pipeline().read("...").chunk_with(...).export_with(JSONPorter(...))
    writes chunks to a temp file without error

# store_in (Handshake step)
test_pipeline_store_in_chroma (mocked)
  - Pipeline().read("...").chunk_with(...).store_in(ChromaHandshake(...))
    calls the handshake write() with the chunks

# from_recipe / from_config
test_pipeline_from_recipe_loads_config
  - Pipeline.from_recipe("default") returns a configured Pipeline instance

test_pipeline_to_config_round_trip
  - pipeline.to_config() -> dict -> Pipeline.from_config(dict)
    produces equivalent pipeline

test_pipeline_describe_returns_dict
  - pipeline.describe() returns a dict with at least one key per step
```

---

### Priority 8 — Porter Base (`src/chonkie/porters/base.py`)

**Missing test:** base porter abstract interface

```
# tests/porters/test_base_porter.py (new file)

test_base_porter_is_abstract
  - Instantiating BasePorter directly raises TypeError

test_base_porter_write_is_abstract
  - Subclass without write() raises TypeError on instantiation

test_base_porter_call_delegates_to_write
  - __call__ invokes write() with same arguments (via concrete subclass)
```

---

## Implementation Order

The following sequencing prioritizes high-impact, low-dependency work first:

1. **Types tests** (Priority 1) — Pure Python, no external deps, high churn risk
2. **Utils tests** (Priority 2) — Filesystem + env var mocking, low dep overhead
3. **Handshake utils tests** (Priority 3) — Trivial to add, standalone function
4. **Cloud refinery tests** (Priority 4) — HTTP mocking with `unittest.mock.patch`
5. **Experimental CodeChunker tests** (Priority 5) — Needs tree-sitter installed
6. **Genie provider tests** (Priority 6) — API mocking, straightforward pattern
7. **Pipeline gap tests** (Priority 7) — Extend existing test file
8. **Porter base tests** (Priority 8) — Small addition, completes porters coverage

---

## Testing Conventions to Follow

All new test files should follow the patterns established in the existing test suite:

- **Fixtures in conftest.py or local to test file** — use `@pytest.fixture` for any reusable setup
- **Skip tests for unavailable optional deps:**
  ```python
  pytest.importorskip("tree_sitter")
  ```
- **Mock external HTTP calls** using `unittest.mock.patch("requests.post")` or `unittest.mock.patch("httpx.post")`
- **Mock file I/O** using `tmp_path` or `monkeypatch.setenv` fixtures
- **Use `pytest.raises`** with `match=` for error message validation
- **Verify index correctness** for all chunkers: `assert original_text[chunk.start_index:chunk.end_index] == chunk.text`
- **No hardcoded API keys** — always mock or use env var monkeypatching

---

## Measuring Progress

After implementing the above, run:

```bash
pytest --cov=src/chonkie --cov-report=term-missing --cov-report=html
```

Target milestones:
- **Phase 1** (Types + Utils + Handshake utils): +8–10% overall coverage
- **Phase 2** (Cloud refineries + Genie providers): +4–6% coverage
- **Phase 3** (Experimental CodeChunker): +6–10% coverage (largest single gap)
- **Phase 4** (Pipeline gaps + Porter base): +2–3% coverage

Estimated total improvement: **+20–30% overall line coverage**
