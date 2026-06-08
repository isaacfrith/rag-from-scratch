# RAG from Scratch

A hands-on project for building **Retrieval-Augmented Generation (RAG)** pipelines with [LangChain](https://python.langchain.com/) and Google Gemini.

The notebooks walk through the full RAG flow — load documents, split them into chunks, embed them into a vector store, retrieve relevant context, and answer questions with an LLM.

## What you're building

```
Web page → chunk → embed → vector store → retrieve → LLM answer
```

| Notebook | What it covers |
|----------|----------------|
| `main.ipynb` | Classic RAG chain with Chroma, Gemini embeddings, and a prompt template |
| `rag_basic.ipynb` | Same pipeline with an in-memory vector store and a LangChain agent with a retrieval tool |
| `rag_query_transformations.ipynb` | Query transformation techniques (work in progress) |

All notebooks use Lilian Weng's [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post as the source document.

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (package manager)
- Python 3.12+
- API keys for Google Gemini (required) and optionally OpenAI, Tavily, and LangSmith

## Setup

```bash
cd rag-from-scratch
uv sync
```

This creates a `.venv` in the project root and installs all dependencies, including Jupyter for notebook support.

### Environment variables

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here   # optional
TAVILY_API_KEY=your-key-here   # optional
USER_AGENT=rag-from-scratch/1.0
```

## Running the notebooks

### In Cursor / VS Code

1. Open a `.ipynb` file.
2. Click the kernel picker (top-right).
3. Select **Python (rag-from-scratch)** or **`.venv (Python 3.13)`** from Python Environments.
4. Run cells with **Shift+Enter**.

Both options use this project's `.venv` — not any other project's environment.

### From the terminal

```bash
uv run jupyter notebook
# or
uv run jupyter lab
```

### Register the kernel (optional)

If **Python (rag-from-scratch)** doesn't appear in the kernel picker:

```bash
uv run python -m ipykernel install \
  --user \
  --name rag-from-scratch \
  --display-name "Python (rag-from-scratch)"
```

## Adding packages

```bash
uv add package-name          # runtime dependency
uv add --dev package-name    # dev-only (e.g. jupyter tools)
```

## Project structure

```
rag-from-scratch/
├── .env                  # API keys (not committed)
├── .venv/                # virtual environment (created by uv sync)
├── main.ipynb            # Chroma-based RAG chain
├── rag_basic.ipynb       # Agent-based RAG with retrieval tool
├── rag_query_transformations.ipynb
├── pyproject.toml        # dependencies
└── uv.lock
```
