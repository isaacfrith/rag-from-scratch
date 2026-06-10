# RAG from Scratch

A hands-on exploration of **Retrieval-Augmented Generation (RAG)** pipelines using [LangChain](https://python.langchain.com/) and Google Gemini — progressing from a bare-bones retrieval chain all the way to advanced indexing and query strategies.

Each notebook is self-contained and teaches a distinct RAG concept, making this a useful reference when learning or evaluating different retrieval approaches.

---

## Notebooks at a glance

| # | Notebook | Topic | Complexity |
|---|----------|-------|------------|
| 1 | [`main.ipynb`](#1-mainipynb--classic-rag-chain) | Classic RAG chain with Chroma | ⭐ Beginner |
| 2 | [`rag_basic.ipynb`](#2-rag_basicipynb--agent-based-rag) | Agent-based RAG with a retrieval tool | ⭐⭐ Beginner–Intermediate |
| 3 | [`rag_query_transformations.ipynb`](#3-rag_query_transformationsipynb--query-transformations) | Multi-query, RAG-Fusion, Decomposition, Step-Back, HyDE | ⭐⭐⭐ Intermediate–Advanced |
| 4 | [`rag_routing.ipynb`](#4-rag_routingipynb--query-routing) | Logical routing & semantic routing | ⭐⭐⭐ Intermediate–Advanced |
| 5 | [`rag_multi_representation_indexing.ipynb`](#5-rag_multi_representation_indexingipynb--multi-representation-indexing) | Summary-based multi-vector indexing | ⭐⭐⭐⭐ Advanced |

All notebooks use Lilian Weng's [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post as their primary source document.

---

## Notebook deep dives

### 1. `main.ipynb` — Classic RAG Chain

> **The foundation.** Load a webpage, split it into chunks, embed and store with Chroma, then retrieve and answer using Gemini.

**Pipeline:**

```mermaid
flowchart LR
    A(["🌐 Web Page"]) --> B["RecursiveCharacter\nTextSplitter"]
    B --> C["Gemini\nEmbeddings"]
    C --> D[("Chroma\nVector Store")]
    D --> E["Similarity\nRetrieval"]
    E --> F["Gemini LLM"]
    F --> G(["✅ Answer"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style D fill:#0369a1,color:#fff,stroke:none
    style F fill:#7c3aed,color:#fff,stroke:none
```

**Key concepts:**
- Document loading with `WebBaseLoader`
- Chunking strategy with `RecursiveCharacterTextSplitter`
- Embedding with `GoogleGenerativeAIEmbeddings`
- Persistent vector storage with `ChromaDB`
- A composable RAG chain using LangChain Expression Language (LCEL)

---

### 2. `rag_basic.ipynb` — Agent-Based RAG

> **Adding agency.** Wraps the retriever as a tool and gives an LLM agent control over when and how to retrieve.

**Pipeline:**

```mermaid
flowchart LR
    A(["❓ Query"]) --> B["ReAct Agent"]
    B -->|"invokes tool"| C["Retrieval Tool"]
    C --> D[("InMemory\nVector Store")]
    D --> E["Retrieved\nContext"]
    E --> B
    B --> F["Gemini LLM"]
    F --> G(["✅ Answer"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style B fill:#7c3aed,color:#fff,stroke:none
    style D fill:#0369a1,color:#fff,stroke:none
    style F fill:#7c3aed,color:#fff,stroke:none
```

**Key concepts:**
- In-memory vector store for fast prototyping
- Wrapping a retriever as a LangChain `Tool`
- Constructing a ReAct-style agent with `create_react_agent`
- Contrasts rule-based retrieval (chain) vs. agent-directed retrieval

---

### 3. `rag_query_transformations.ipynb` — Query Transformations

> **The biggest notebook.** Five distinct techniques that transform the user's query before (or during) retrieval to improve answer quality.

#### 3a. Multi-Query
The original question is rewritten into **multiple alternative phrasings** using an LLM. Each variant is sent to the retriever independently, and the union of results provides richer context.

```mermaid
flowchart LR
    A(["❓ Question"]) --> B["LLM: Query\nRewriter"]
    B --> Q1["Variant 1"]
    B --> Q2["Variant 2"]
    B --> Q3["Variant N"]
    Q1 & Q2 & Q3 --> R[("Vector Store")]
    R --> D["Deduplicate\nDocs"]
    D --> E["LLM"]
    E --> G(["✅ Answer"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style R fill:#0369a1,color:#fff,stroke:none
    style E fill:#7c3aed,color:#fff,stroke:none
```

#### 3b. RAG-Fusion
Extends multi-query by applying **Reciprocal Rank Fusion (RRF)** to re-rank the retrieved documents across all query variants before passing context to the LLM — producing a single, high-quality ranked list.

```mermaid
flowchart LR
    A(["❓ Question"]) --> B["LLM: Query\nRewriter"]
    B --> Q1["Query 1"] & Q2["Query 2"] & Q3["Query N"]
    Q1 --> R1["Ranked\nResults 1"]
    Q2 --> R2["Ranked\nResults 2"]
    Q3 --> R3["Ranked\nResults N"]
    R1 & R2 & R3 --> RRF["Reciprocal\nRank Fusion"]
    RRF --> D["Top-K Docs"]
    D --> E["LLM"]
    E --> G(["✅ Answer"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style RRF fill:#b45309,color:#fff,stroke:none
    style E fill:#7c3aed,color:#fff,stroke:none
```

#### 3c. Decomposition
The question is broken down into **independent sub-questions**. Each sub-question is answered sequentially, with prior Q&A pairs fed as context to later sub-questions — effectively chaining knowledge.

```mermaid
flowchart TD
    A(["❓ Question"]) --> B["LLM: Decompose"]
    B --> Q1["Sub-Q 1"] & Q2["Sub-Q 2"] & Q3["Sub-Q 3"]
    Q1 -->|retrieve + answer| P1["Q&A Pair 1"]
    P1 -->|context| Q2
    Q2 -->|retrieve + answer| P2["Q&A Pair 2"]
    P2 -->|context| Q3
    Q3 -->|retrieve + answer| P3["Q&A Pair 3"]
    P3 --> F["LLM: Synthesise"]
    F --> G(["✅ Final Answer"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style B fill:#7c3aed,color:#fff,stroke:none
    style F fill:#7c3aed,color:#fff,stroke:none
```

#### 3d. Step-Back Prompting
Uses few-shot examples to generate a **more abstract, generalised version** of the question. Both the original and step-back questions are used for retrieval, combining specific and broad context.

```mermaid
flowchart LR
    A(["❓ Question"]) --> R1[("Vector Store")]
    A --> SB["LLM: Step-Back\n(few-shot)"]
    SB --> AQ["Abstract Question"]
    AQ --> R1
    R1 --> MC["Merged\nContext"]
    MC --> E["LLM"]
    E --> G(["✅ Answer"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style SB fill:#b45309,color:#fff,stroke:none
    style R1 fill:#0369a1,color:#fff,stroke:none
    style E fill:#7c3aed,color:#fff,stroke:none
```

#### 3e. HyDE (Hypothetical Document Embeddings)
Instead of embedding the raw question, the LLM first **generates a hypothetical answer passage**. That passage is embedded and used for retrieval — exploiting the semantic proximity between a "fake" answer and real documents.

```mermaid
flowchart LR
    A(["❓ Question"]) --> B["LLM: Generate\nHypothetical Passage"]
    B --> C["Embed\nPassage"]
    C --> D[("Vector Store")]
    D --> E["Retrieved\nReal Docs"]
    E --> F["LLM"]
    F --> G(["✅ Answer"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style B fill:#b45309,color:#fff,stroke:none
    style D fill:#0369a1,color:#fff,stroke:none
    style F fill:#7c3aed,color:#fff,stroke:none
```

---

### 4. `rag_routing.ipynb` — Query Routing

> **Directing traffic.** Decides which data source or retrieval strategy to use based on the nature of the query.

#### 4a. Logical Routing (Structured Output)
Uses an LLM with structured output (Pydantic schema) to classify the query and route it to the correct data source (e.g. Python docs vs. JavaScript docs vs. Go docs).

```mermaid
flowchart LR
    A(["❓ Question"]) --> B["LLM + RouteQuery\nPydantic Schema"]
    B -->|"python_docs"| C["Python Docs\nRetriever"]
    B -->|"js_docs"| D["JS Docs\nRetriever"]
    B -->|"golang_docs"| E["Go Docs\nRetriever"]
    C & D & E --> F["LLM"]
    F --> G(["✅ Answer"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style B fill:#b45309,color:#fff,stroke:none
    style F fill:#7c3aed,color:#fff,stroke:none
```

Also demonstrates a fallback approach using `PydanticOutputParser` for models that don't support native tool-calling / structured output.

#### 4b. Semantic Routing
Embeds the query and compares it via **cosine similarity** against a set of pre-embedded prompt templates. The most semantically similar prompt is selected and used for the response.

```mermaid
flowchart LR
    A(["❓ Question"]) --> E["Embed Query"]
    P1["Physics Prompt"] --> EP1["Embedded"]
    P2["Math Prompt"] --> EP2["Embedded"]
    E --> COS["Cosine Similarity"]
    EP1 & EP2 --> COS
    COS -->|"highest match"| SEL["Selected Prompt"]
    SEL --> F["LLM"]
    F --> G(["✅ Answer"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style COS fill:#b45309,color:#fff,stroke:none
    style F fill:#7c3aed,color:#fff,stroke:none
```

#### 4c. Query Structuring
Converts a free-text question into a **structured database query** using a Pydantic schema (`TutorialSearch`). Demonstrates generating typed filters like `min_view_count`, `max_view_count`, and `publish_date_range` from natural language.

```mermaid
flowchart LR
    A(["📝 Natural Language Query"]) --> B["LLM + TutorialSearch\nPydantic Schema"]
    B --> C1["title_search"]
    B --> C2["max_view_count"]
    B --> C3["publish_date_range"]
    C1 & C2 & C3 --> D["Structured DB Query"]
    D --> G(["✅ Filtered Results"])

    style A fill:#4f46e5,color:#fff,stroke:none
    style G fill:#059669,color:#fff,stroke:none
    style B fill:#b45309,color:#fff,stroke:none
```

---

### 5. `rag_multi_representation_indexing.ipynb` — Multi-Representation Indexing

> **Smarter indexing.** Index compact summaries for retrieval, but return the full parent documents to the LLM.

**Pipeline:**

```mermaid
flowchart TB
    subgraph INDEX ["📥 Indexing"]
        direction LR
        D(["Full Docs"]) --> S["LLM: Summarise"]
        S --> SE["Embed Summaries"]
        SE --> VS[("Chroma\nVector Store")]
        D --> BS[("InMemory\nByte Store")]
    end

    subgraph RETRIEVE ["🔍 Retrieval"]
        direction LR
        Q(["❓ Query"]) --> MVR["MultiVectorRetriever"]
        MVR -->|"search summaries"| VS
        VS --> MVR
        MVR -->|"fetch by doc_id"| BS
        BS --> F["LLM"]
        F --> A(["✅ Answer"])
    end

    INDEX -.-> RETRIEVE

    style D fill:#4f46e5,color:#fff,stroke:none
    style Q fill:#4f46e5,color:#fff,stroke:none
    style A fill:#059669,color:#fff,stroke:none
    style VS fill:#0369a1,color:#fff,stroke:none
    style BS fill:#0369a1,color:#fff,stroke:none
    style MVR fill:#b45309,color:#fff,stroke:none
    style F fill:#7c3aed,color:#fff,stroke:none
```

**Key concepts:**
- `MultiVectorRetriever` from LangChain — decouples what you *index* from what you *retrieve*
- Summary-based indexing: summaries are more semantically dense than raw chunks
- Batch LLM calls for summarisation with `max_concurrency`
- Parent-child document store architecture
- Sources: Lilian Weng's [LLM Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) and [Human Data Quality](https://lilianweng.github.io/posts/2024-02-05-human-data-quality/) posts

---

## Tech stack

| Library | Purpose |
|---------|---------|
| `langchain` / `langchain-core` | Chains, prompts, LCEL |
| `langchain-google-genai` | Gemini LLM + embeddings |
| `langchain-openai` | Local LLM via LM Studio (OpenAI-compatible API) |
| `langchain-community` | Web loaders, YouTube loader, Chroma, math utils |
| `langchain-huggingface` | HuggingFace sentence-transformer embeddings |
| `langchain-classic` | `MultiVectorRetriever` |
| `chromadb` | Persistent vector store |
| `sentence-transformers` | Local embedding models |
| `langsmith` | Tracing & prompt hub |
| `pydantic` | Structured output schemas |
| `umap-learn` / `plotly` | Embedding visualisation |

---

## Prerequisites

- [uv](https://docs.astral.sh/uv/) (package manager)
- Python 3.12+
- Google Gemini API key (required)
- OpenAI API key or a local LM Studio server on `http://127.0.0.1:1234` (used in several notebooks)
- Optional: Tavily API key, LangSmith API key

---

## Setup

```bash
git clone https://github.com/your-username/rag-from-scratch.git
cd rag-from-scratch
uv sync
```

Create a `.env` file in the project root:

```bash
GOOGLE_API_KEY=your-gemini-key
OPENAI_API_KEY=your-openai-key   # or any value if using LM Studio
TAVILY_API_KEY=your-key          # optional
LANGSMITH_API_KEY=your-key       # optional
USER_AGENT=rag-from-scratch/1.0
```

---

## Running the notebooks

### In Cursor / VS Code

1. Open a `.ipynb` file.
2. Click the kernel picker (top-right).
3. Select **Python (rag-from-scratch)** or **`.venv (Python 3.13)`** from Python Environments.
4. Run cells with **Shift+Enter**.

### From the terminal

```bash
uv run jupyter lab
# or
uv run jupyter notebook
```

### Register the kernel (if it doesn't appear)

```bash
uv run python -m ipykernel install \
  --user \
  --name rag-from-scratch \
  --display-name "Python (rag-from-scratch)"
```

---

## Project structure

```
rag-from-scratch/
├── .env                                      # API keys (not committed)
├── .venv/                                    # virtual environment (uv sync)
├── main.ipynb                                # 1. Classic RAG chain
├── rag_basic.ipynb                           # 2. Agent-based RAG
├── rag_query_transformations.ipynb           # 3. Multi-query, Fusion, Decomposition, Step-Back, HyDE
├── rag_routing.ipynb                         # 4. Logical, semantic & structured routing
├── rag_multi_representation_indexing.ipynb   # 5. Summary-indexed multi-vector retrieval
├── pyproject.toml
└── uv.lock
```

---

## Adding packages

```bash
uv add package-name          # runtime dependency
uv add --dev package-name    # dev-only (e.g. jupyter tools)
```
