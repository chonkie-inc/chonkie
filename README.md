<div align='center'>

![Chonkie Logo](https://github.com/chonkie-inc/chonkie/blob/main/assets/chonkie_logo_br_transparent_bg.png?raw=true)

# 🦛 Chonkie ✨

[![PyPI version](https://img.shields.io/pypi/v/chonkie.svg)](https://pypi.org/project/chonkie/)
[![License](https://img.shields.io/github/license/chonkie-inc/chonkie.svg)](https://github.com/chonkie-inc/chonkie/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-chonkie.ai-blue.svg)](https://docs.chonkie.ai)
[![Package size](https://img.shields.io/badge/size-450KB-blue)](https://github.com/chonkie-inc/chonkie/blob/main/README.md#installation)
[![codecov](https://codecov.io/gh/chonkie-inc/chonkie/graph/badge.svg?token=V4EWIJWREZ)](https://codecov.io/gh/chonkie-inc/chonkie)
[![Downloads](https://static.pepy.tech/badge/chonkie)](https://pepy.tech/project/chonkie)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/vH3SkRqmUz?style=flat)](https://discord.gg/vH3SkRqmUz)
[![GitHub stars](https://img.shields.io/github/stars/chonkie-inc/chonkie.svg)](https://github.com/chonkie-inc/chonkie/stargazers)

_The no-nonsense ultra-light and lightning-fast chunking library that's ready to CHONK your texts!_

[Installation](#installation) •
[Usage](#basic-usage) •
[Pipeline](#the-chonkie-pipeline) •
[Chunkers](#chunkers) •
[Integrations](#integrations) •
[Benchmarks](#benchmarks)

</div>

Tired of making your gazillionth chunker? Sick of the overhead of large libraries? Want to chunk your texts quickly and efficiently? Chonkie the mighty hippo is here to help!

**🚀 Feature-rich**: All the CHONKs you'd ever need </br>
**✨ Easy to use**: Install, Import, CHONK </br>
**⚡ Fast**: CHONK at the speed of light! zooooom </br>
**🪶 Light-weight**: No bloat, just CHONK </br>
**🌏 Wide support**: CHONKie [integrates](#integrations) with your favorite tokenizer, embedding model and APIs! </br>
**💬 ️Multilingual**: Out-of-the-box support for 56 languages </br>
**☁️ Cloud-Ready**: CHONK locally or in the [Chonkie Cloud](https://cloud.chonkie.ai) </br>
**🦛 Cute CHONK mascot**: psst it's a pygmy hippo btw </br>
**❤️ [Moto Moto](#acknowledgements)'s favorite python library** </br>

**Chonkie** is a chunking library that "**just works**" ✨

## Installation

To install chonkie, run:

```bash
pip install chonkie
```

Chonkie follows the rule of minimum installs.
Have a favorite chunker? Read our [docs](https://docs.chonkie.ai) to install only what you need
Don't want to think about it? Simply install `all` (Not recommended for production environments)

```bash
pip install chonkie[all]
```

## Basic Usage

Here's a basic example to get you started:

```python
# First import the chunker you want from Chonkie
from chonkie import RecursiveChunker

# Initialize the chunker
chunker = RecursiveChunker()

# Chunk some text
chunks = chunker("Chonkie is the goodest boi! My favorite chunking hippo hehe.")

# Access chunks
for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")
```

Check out more usage examples in the [docs](https://docs.chonkie.ai)!

## The Chonkie Pipeline

Chonkie processes text using a pipeline approach to transform raw documents into refined, usable chunks. This allows for flexibility and efficiency in handling different chunking strategies. We call this pipeline `CHOMP` (short for _'CHOnkie's Multi-step Pipeline'_).

Here's a conceptual overview of the pipeline, as illustrated in the diagram:

![🤖 CHOMP pipeline diagram](./assets/chomp-transparent-bg.png)

The main stages are:

1. **📄 Document**: The starting point – your input text data. It can be in any format!
2. **👨‍🍳 Chef**: This stage handles initial text preprocessing. It might involve cleaning, normalization, or other preparatory steps to get the text ready for chunking. While this is optional, it is recommended to use the `Chef` stage to clean your text before chunking.
3. **🦛 Chunker**: The core component you select (e.g., RecursiveChunker, SentenceChunker). It applies its specific logic to split the preprocessed text into initial chunks based on the chosen strategy and parameters.
4. **🏭 Refinery**: After initial chunking, the Refinery performs post-processing. This can include merging small chunks based on overlap, adding embeddings, or adding additional context to the chunks. It helps ensure the quality and consistency of the output. You can have multiple `Refineries` to apply different post-processing steps.
5. **🤗 Friends**: The pipeline's produces the final results which can be either exported to be saved or ingested into your vector database. Chonkie offers `Porters` to export the chunks and `Handshakes` to ingest the chunks into your vector database.
   - **🐴 Porters**: Porters can save the chunks to a file or a database. Currently, only `JSON` is supported for exporting the chunks.
   - **🤝 Handshakes**: Handshakes provide a unified interface for ingesting the chunks into your preferred vector databases.

This modular pipeline allows Chonkie to be both powerful and easy to configure for various text chunking needs.

## Chunkers

Chonkie provides several chunkers to help you split your text efficiently for RAG applications. Here's a quick overview of the available chunkers:

| Name               | Alias       | Description                                                                                                                |
| ------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------- |
| `TokenChunker`     | `token`     | Splits text into fixed-size token chunks.                                                                                  |
| `SentenceChunker`  | `sentence`  | Splits text into chunks based on sentences.                                                                                |
| `RecursiveChunker` | `recursive` | Splits text hierarchically using customizable rules to create semantically meaningful chunks.                              |
| `SemanticChunker`  | `semantic`  | Splits text into chunks based on semantic similarity. Inspired by the work of [Greg Kamradt](https://github.com/gkamradt). |
| `LateChunker`      | `late`      | Embeds text and then splits it to have better chunk embeddings.                                                            |
| `CodeChunker`      | `code`      | Splits code into structurally meaningful chunks.                                                                           |
| `NeuralChunker`    | `neural`    | Splits text using a neural model.                                                                                          |
| `SlumberChunker`   | `slumber`   | Splits text using an LLM to find semantically meaningful chunks. Also known as _"AgenticChunker"_.                         |

More on these methods and the approaches taken inside the [docs](https://docs.chonkie.ai)

## Integrations

Chonkie boasts 24+ integrations across tokenizers, embedding providers, LLMs, refineries, porters, vector databases, and utilities, ensuring it fits seamlessly into your existing workflow.

<details>
<summary><strong>🪓 Slice 'n' Dice! Chonkie supports 5+ ways to tokenize! </strong></summary>

Choose from supported tokenizers or provide your own custom token counting function. Flexibility first!

| Name           | Description                                                    | Optional Install      |
| -------------- | -------------------------------------------------------------- | --------------------- |
| `character`    | Basic character-level tokenizer. **Default tokenizer.**        | `default`             |
| `word`         | Basic word-level tokenizer.                                    | `default`             |
| `tokenizers`   | Load any tokenizer from the Hugging Face `tokenizers` library. | `chonkie[tokenizers]` |
| `tiktoken`     | Use OpenAI's `tiktoken` library (e.g., for `gpt-4`).           | `chonkie[tiktoken]`   |
| `transformers` | Load tokenizers via `AutoTokenizer` from HF `transformers`.    | `chonkie[neural]`     |

`default` indicates that the feature is available with the default `pip install chonkie`.

To use a custom token counter, you can pass in any function that takes a string and returns an integer! Something like this:

```python
def custom_token_counter(text: str) -> int:
    return len(text)

chunker = RecursiveChunker(tokenizer_or_token_counter=custom_token_counter)
```

You can use this to extend Chonkie to support any tokenization scheme you want!

</details>

<details>
<summary><strong>🧠 Embed like a boss! Chonkie links up with 8+ embedding pals!</strong></summary>

Seamlessly works with various embedding model providers. Bring your favorite embeddings to the CHONK party! Use `AutoEmbeddings` to load models easily.

| Provider / Alias        | Class                           | Description                            | Optional Install        |
| ----------------------- | ------------------------------- | -------------------------------------- | ----------------------- |
| `model2vec`             | `Model2VecEmbeddings`           | Use `Model2Vec` models.                | `chonkie[model2vec]`    |
| `sentence-transformers` | `SentenceTransformerEmbeddings` | Use any `sentence-transformers` model. | `chonkie[st]`           |
| `openai`                | `OpenAIEmbeddings`              | Use OpenAI's embedding API.            | `chonkie[openai]`       |
| `azure-openai`          | `AzureOpenAIEmbeddings`         | Use Azure OpenAI embedding service.    | `chonkie[azure-openai]` |
| `cohere`                | `CohereEmbeddings`              | Use Cohere's embedding API.            | `chonkie[cohere]`       |
| `gemini`                | `GeminiEmbeddings`              | Use Google's Gemini embedding API.     | `chonkie[gemini]`       |
| `jina`                  | `JinaEmbeddings`                | Use Jina AI's embedding API.           | `chonkie[jina]`         |
| `voyageai`              | `VoyageAIEmbeddings`            | Use Voyage AI's embedding API.         | `chonkie[voyageai]`     |

</details>

<details>
<summary><strong>🧞‍♂️ Power Up with Genies! Chonkie supports 3+ LLM providers!</strong></summary>

Genies provide interfaces to interact with Large Language Models (LLMs) for advanced chunking strategies or other tasks within the pipeline.

| Genie Name     | Class              | Description                       | Optional Install        |
| -------------- | ------------------ | --------------------------------- | ----------------------- |
| `gemini`       | `GeminiGenie`      | Interact with Google Gemini APIs. | `chonkie[gemini]`       |
| `openai`       | `OpenAIGenie`      | Interact with OpenAI APIs.        | `chonkie[openai]`       |
| `azure-openai` | `AzureOpenAIGenie` | Interact with Azure OpenAI APIs.  | `chonkie[azure-openai]` |

You can also use the `OpenAIGenie` to interact with any LLM provider that supports the OpenAI API format, by simply changing the `model`, `base_url`, and `api_key` parameters. For example, here's how to use the `OpenAIGenie` to interact with the `Llama-4-Maverick` model via OpenRouter:

```python
from chonkie import OpenAIGenie

genie = OpenAIGenie(model="meta-llama/llama-4-maverick",
                    base_url="https://openrouter.ai/api/v1",
                    api_key="your_api_key")
```

</details>

<details>
<summary><strong>🏭 Refine your CHONKs! Chonkie supports 2+ refineries!</strong></summary>

Refineries help you post-process and enhance your chunks after initial chunking.

| Refinery Name | Class                | Description                                   | Optional Install    |
| ------------- | -------------------- | --------------------------------------------- | ------------------- |
| `overlap`     | `OverlapRefinery`    | Merge overlapping chunks based on similarity. | `default`           |
| `embeddings`  | `EmbeddingsRefinery` | Add embeddings to chunks using any provider.  | `chonkie[semantic]` |

</details>

<details>
<summary><strong>🐴 Exporting CHONKs! Chonkie supports 2+ Porters!</strong></summary>

Porters help you save your chunks easily.

| Porter Name | Class            | Description                            | Optional Install    |
| ----------- | ---------------- | -------------------------------------- | ------------------- |
| `json`      | `JSONPorter`     | Export chunks to a JSON file.          | `default`           |
| `datasets`  | `DatasetsPorter` | Export chunks to HuggingFace datasets. | `chonkie[datasets]` |

</details>

<details>
<summary><strong>🤝 Shake hands with your DB! Chonkie connects with 4+ vector stores!</strong></summary>

Handshakes provide a unified interface to ingest chunks directly into your favorite vector databases.

| Handshake Name | Class                  | Description                                  | Optional Install    |
| -------------- | ---------------------- | -------------------------------------------- | ------------------- |
| `chroma`       | `ChromaHandshake`      | Ingest chunks into ChromaDB.                 | `chonkie[chroma]`   |
| `qdrant`       | `QdrantHandshake`      | Ingest chunks into Qdrant.                   | `chonkie[qdrant]`   |
| `pgvector`     | `PgvectorHandshake`    | Ingest chunks into PostgreSQL with pgvector. | `chonkie[pgvector]` |
| `turbopuffer`  | `TurbopufferHandshake` | Ingest chunks into Turbopuffer.              | `chonkie[tpuf]`     |
| `pinecone`     | `PineconeHandshake`    | Ingest chunks into Pinecone.                 | `chonkie[pinecone]` |
| `weaviate`     | `WeaviateHandshake`    | Ingest chunks into Weaviate.                 | `chonkie[weaviate]` |
| `mongodb`      | `MongoDBHandshake`     | Ingest chunks into MongoDB.                  | `chonkie[mongodb]`  |

</details>

<details>
<summary><strong>🛠️ Utilities & Helpers! Chonkie includes handy tools!</strong></summary>

Additional utilities to enhance your chunking workflow.

| Utility Name | Class        | Description                                    | Optional Install |
| ------------ | ------------ | ---------------------------------------------- | ---------------- |
| `hub`        | `Hubbie`     | Simple wrapper for HuggingFace Hub operations. | `chonkie[hub]`   |
| `viz`        | `Visualizer` | Rich console visualizations for chunks.        | `chonkie[viz]`   |

</details>

<details>
<summary><strong>👨‍🍳 Chefs & 📁 Fetchers! Text preprocessing and data loading!</strong></summary>

Chefs handle text preprocessing, while Fetchers load data from various sources.

| Component | Class         | Description                           | Optional Install |
| --------- | ------------- | ------------------------------------- | ---------------- |
| `chef`    | `TextChef`    | Text preprocessing and cleaning.      | `default`        |
| `fetcher` | `FileFetcher` | Load text from files and directories. | `default`        |

</details>

With Chonkie's wide range of integrations, you can easily plug it into your existing infrastructure and start CHONKING!

## Benchmarks

> "I may be smol hippo, but I pack a big punch!" 🦛

Chonkie is not just cute, it's also fast and efficient! Here's how it stacks up against the competition:

**Size**📦

- **Default Install:** 15MB (vs 80-171MB for alternatives)
- **With Semantic:** Still 10x lighter than the closest competition!

**Speed**⚡

- **Token Chunking:** 33x faster than the slowest alternative
- **Sentence Chunking:** Almost 2x faster than competitors
- **Semantic Chunking:** Up to 2.5x faster than others

Check out our detailed [benchmarks](BENCHMARKS.md) to see how Chonkie races past the competition! 🏃‍♂️💨

## Contributing

Want to help grow Chonkie? Check out [CONTRIBUTING.md](CONTRIBUTING.md) to get started! Whether you're fixing bugs, adding features, or improving docs, every contribution helps make Chonkie a better CHONK for everyone.

Remember: No contribution is too small for this tiny hippo! 🦛

## Acknowledgements

Chonkie would like to CHONK its way through a special thanks to all the users and contributors who have helped make this library what it is today! Your feedback, issue reports, and improvements have helped make Chonkie the CHONKIEST it can be.

And of course, special thanks to [Moto Moto](https://www.youtube.com/watch?v=I0zZC4wtqDQ&t=5s) for endorsing Chonkie with his famous quote:

> "I like them big, I like them chonkie." ~ Moto Moto

## Citation

If you use Chonkie in your research, please cite it as follows:

```bibtex
@software{chonkie2025,
  author = {Minhas, Bhavnick AND Nigam, Shreyash},
  title = {Chonkie: A no-nonsense fast, lightweight, and efficient text chunking library},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/chonkie-inc/chonkie}},
}
```
