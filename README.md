<div align='center'>

![Chonkie Logo](https://github.com/chonkie-inc/chonkie/blob/main/assets/chonkie_logo_br_transparent_bg.png)

# 🦛 Chonkie ✨

[![PyPI version](https://img.shields.io/pypi/v/chonkie.svg)](https://pypi.org/project/chonkie/)
[![License](https://img.shields.io/github/license/chonkie-inc/chonkie.svg)](https://github.com/chonkie-inc/chonkie/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-chonkie.ai-blue.svg)](https://docs.chonkie.ai)
[![Package size](https://img.shields.io/badge/size-15MB-blue)](https://github.com/chonkie-inc/chonkie/blob/main/README.md#installation)
[![codecov](https://codecov.io/gh/chonkie-inc/chonkie/graph/badge.svg?token=V4EWIJWREZ)](https://codecov.io/gh/chonkie-inc/chonkie)
[![Downloads](https://static.pepy.tech/badge/chonkie)](https://pepy.tech/project/chonkie)
[![Discord](https://dcbadge.limes.pink/api/server/https://discord.gg/rYYp6DC4cv?style=flat)](https://discord.gg/rYYp6DC4cv)
[![GitHub stars](https://img.shields.io/github/stars/chonkie-inc/chonkie.svg)](https://github.com/chonkie-inc/chonkie/stargazers)

_The no-nonsense ultra-light and lightning-fast chunking library that's ready to CHONK your texts!_

[Installation](#installation) •
[Usage](#usage) •
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
**💬 ️Multilingual**: Out-of-the-box support for 5+ language CHONKS (more coming 🔜) </br>
**☁️ Cloud-Ready**: CHONK locally or in the [Chonkie Cloud](https://cloud.chonkie.ai) </br>
**🦛 Cute CHONK mascot**: psst it's a pygmy hippo btw </br>
**❤️ [Moto Moto](#acknowledgements)'s favorite python library** </br>

**Chonkie** is a chunking library that "**just works**" ✨

# Installation

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

# Usage

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

# Chunkers

Chonkie provides several chunkers to help you split your text efficiently for RAG applications. Here's a quick overview of the available chunkers:

- **TokenChunker**: Splits text into fixed-size token chunks.
- **SentenceChunker**: Splits text into chunks based on sentences.
- **RecursiveChunker**: Splits text hierarchically using customizable rules to create semantically meaningful chunks.
- **SemanticChunker**: Splits text into chunks based on semantic similarity.
- **SDPMChunker**: Splits text using a Semantic Double-Pass Merge approach.
- **LateChunker**: Embeds text and then splits it to have better chunk embeddings.
- **CodeChunker**: Splits code into structurally meaningful chunks.
- **NeuralChunker**: Splits text using a neural model.
- **SlumberChunker**: Splits text using an LLM to find semantically meaningful chunks. Also known as _"AgenticChunker"_.

More on these methods and the approaches taken inside the [docs](https://docs.chonkie.ai)

# Integrations

Chonkie integrates smoothly with the tools you already use:

- **Tokenizers:** Choose from **3 supported tokenizers** (like `Hugging Face 🤗` and `Tiktoken`) or provide your own custom token counting function. Flexibility first!
- **Embedding Models:** Seamlessly works with **5 out-of-the-box embedding model providers**, including `SentenceTransformers`, `Model2Vec`, `OpenAI`, `Cohere`, and `Jina AI`. Bring your favorite embeddings to the CHONK party!

# Benchmarks

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

# Contributing

Want to help grow Chonkie? Check out [CONTRIBUTING.md](CONTRIBUTING.md) to get started! Whether you're fixing bugs, adding features, or improving docs, every contribution helps make Chonkie a better CHONK for everyone.

Remember: No contribution is too small for this tiny hippo! 🦛

# Acknowledgements

Chonkie would like to CHONK its way through a special thanks to all the users and contributors who have helped make this library what it is today! Your feedback, issue reports, and improvements have helped make Chonkie the CHONKIEST it can be.

And of course, special thanks to [Moto Moto](https://www.youtube.com/watch?v=I0zZC4wtqDQ&t=5s) for endorsing Chonkie with his famous quote:
> "I like them big, I like them chonkie." ~ Moto Moto

# Citation

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
