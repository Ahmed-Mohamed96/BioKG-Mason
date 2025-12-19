<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/80288506-9d8c-4aa8-a387-cbc94b0698c0" />

# BioKG-Mason – Biomedical Knowledge Graph Construction with LLM Agents

**BioKG-Mason** builds a **biomedical knowledge graph (KG)** from PDF articles named by PMID, using a pipeline of LLM-based agents plus Neo4j.

Given:

- One or more biomedical **PDF files** (filenames = PMIDs)  
- An **(initially empty) Neo4j database**

BioKG-Mason will:

1. Extract **paragraphs and tables** from each PDF.
2. Use LLM Agents to extract, process, and store **biomedical triplets** from each text segment in Neo4j with **full provenance**:
   - `pmid`
   - `source_text` (paragraph or flattened table text)
   - `source_kind` (`paragraph` or `table`)
   - `page` and `table_index` (for tables)

The project supports:

- **LLM backends**: OpenAI (via `ChatOpenAI`) or Ollama (`ChatOllama`).
- **Embeddings backends**:
  - OpenAI embeddings API, or
  - Local `sentence-transformers` models when embeddings provider is set to `ollama`.

---

## 1. Features

- **PDF ingestion**
  - Extracts **paragraphs** and **tables** from PDFs using `pdfplumber`.
  - Tables are flattened into text with column/value hints for the LLM.

- **Triplet extraction**
  - LLM-based `TripletExtractor` (LangChain structured output).
  - Uses `ChatOpenAI` or `ChatOllama` with `temperature=0.0` for deterministic triplet extraction.

- **Entity resolution**
  - `EntityMatcher` uses embeddings (OpenAI or sentence-transformers) + cosine similarity against Neo4j nodes.
  - Avoids duplicate nodes for the same biomedical entity.

- **LLM-based entity match validation**
  - `EntityMatchValidator` checks each embedding-based match in context of the paragraph/table.
  - If the LLM rejects the match, a **new node** is created instead.

- **LLM-based schema normalization**
  - `SchemaMatcher` uses an LLM to assign:
    - normalized **node labels** (e.g., `PROTEIN`, `GENE`, `DISEASE`, `PATHWAY`)
    - normalized **relationship types** (e.g., `ACTIVATES`, `INHIBITS`, `ASSOCIATED_WITH`)
  - Always sees a **seed schema** from `schema.json` plus labels/types discovered in Neo4j.
  - Keeps a cache in memory for the duration of the run (no on-disk updates).

- **Neo4j model**
  - Nodes:
    - labels: biomedical categories (`PROTEIN`, `GENE`, `RNA`, `DISEASE`, ...)
    - properties: `entity_id` (UUID), `name`, `embedding` (vector).
  - Relationships:
    - types: normalized types (`ACTIVATES`, `MODULATES`, `ASSOCIATED_WITH`, etc.).
    - properties: `pmid`, `source_text`, `source_kind`, `page`, `table_index`.

---

## 2. Repository Layout

```text
BioKG-Mason/
  pyproject.toml
  config.yml
  schema.json
  src/
    biokg_agents/
      __init__.py
      config.py
      pipeline.py
      modules/
        __init__.py
        base.py
        triplets.py
        triplet_extractor.py
        entity_matcher.py
        entity_match_validator.py
        schema_matcher.py
        pdf_reader.py
        neo4j_client.py
        openai_client.py
        sentence_transformers_client.py
        ollama_client.py
  pdfs/
    12345678.pdf (example)
    87654321.pdf (example)
```

Key package: `biokg_agents` (under `src/`).

---

## 3. Requirements

### 3.1. Software

- Python 3.9+  
- [PDM](https://pdm.fming.dev/latest/) (Python package/dependency manager)
- Neo4j 4.x / 5.x (local or remote)
- For OpenAI:
  - OpenAI account & API key
- For Ollama:
  - [Ollama](https://ollama.com/) installed and running (if using local LLM)
- For local embeddings:
  - `sentence-transformers` and optional HF model dependencies (e.g., `einops` for some models)

### 3.2. Python dependencies

Already managed via PDM in `pyproject.toml`, but conceptually include:

- `langchain`, `langchain-openai`, `langchain-community`
- `openai`
- `sentence-transformers`
- `neo4j`
- `pdfplumber`
- `PyYAML`
- `requests`
- `pydantic`
- `einops` (for some Hugging Face models, e.g. Nomic)

---

## 4. Cloning and Installation

### 4.1. Clone the repository

If you have SSH access:

```bash
git clone git@github.com:Ahmed-Mohamed96/BioKG-Mason.git
cd BioKG-Mason
```

If SSH fails, use HTTPS:

```bash
git clone https://github.com/Ahmed-Mohamed96/BioKG-Mason.git
cd BioKG-Mason
```

### 4.2. Install dependencies with PDM (Pre-installation of PDM is required)

From the project root:

```bash
pdm install
```

PDM will:

- Create a project-specific virtual environment.
- Install all declared dependencies into it.

To verify the environment, run:

```bash
pdm run python -c "import biokg_agents; print(biokg_agents.__file__)"
```

You should see a path under `src/biokg_agents/__init__.py`.

---

## 5. Configuration

The pipeline reads settings from `config.yml` (in the project root by default).

### 5.1. Example `config.yml`

```yaml
llm:
  provider: openai        # "openai" or "ollama"
  model: gpt-4o-mini      # OpenAI model name (or Ollama model if provider=ollama)
  temperature: 0.0

embeddings:
  provider: ollama        # "openai" or "ollama"
  # If provider=openai: use an OpenAI embedding model name
  # If provider=ollama: use a sentence-transformers / HF model name
  model: sentence-transformers/all-mpnet-base-v2

neo4j:
  uri: neo4j://localhost:7687
  user: neo4j
  password: "your_password"     # ensure this is quoted to stay a string
  database: neo4j               # or your named database

pdf:
  input_dir: ./pdfs             # folder containing PDF files named by PMID

pipeline:
  similarity_threshold: 0.82    # cosine similarity threshold for entity matching

schema:
  file: schema.json             # initial node labels & relationship types
```

### 5.2. OpenAI configuration

Set your API key in the environment:

```bash
export OPENAI_API_KEY="sk-REPLACE_WITH_YOUR_KEY"
```

`ChatOpenAI` and `OpenAIEmbeddingsClient` will pick it up automatically.

### 5.3. Ollama configuration (for LLM)

If you use Ollama as the LLM:

```yaml
llm:
  provider: ollama
  model: llama3
  temperature: 0.0
```

Ensure:

```bash
ollama serve
ollama pull llama3
```

### 5.4. Local sentence-transformers embeddings

When `embeddings.provider: ollama`, the code uses **sentence-transformers** locally, not Ollama’s embedding API. In this case:

```yaml
embeddings:
  provider: ollama
  model: sentence-transformers/all-mpnet-base-v2
```

On first run, this will download the model from Hugging Face (internet required). Some HF models (e.g., Nomic) may require extra packages like `einops`.

---

## 6. Initial Schema (`schema.json`)

`schema.json` seeds the schema aligner with preferred node labels and relationship types. These are used by the LLM-based schema matcher on **every** triplet (in addition to labels/types already in Neo4j).

Example `schema.json`:

```json
{
  "node_labels": [
    "PROTEIN",
    "GENE",
    "RNA",
    "MICRO_RNA",
    "DNA",
    "DRUG",
    "COMPOUND",
    "METABOLITE",
    "DISEASE",
    "PHENOTYPE",
    "SYMPTOM",
    "BIOMARKER",
    "CELL",
    "CELL_LINE",
    "TISSUE",
    "ORGAN",
    "ORGANISM",
    "STRAIN",
    "VARIANT",
    "PATHWAY",
    "PROCESS"
  ],
  "relationship_types": [
    "ACTIVATES",
    "INHIBITS",
    "BINDS_TO",
    "INTERACTS_WITH",
    "UPREGULATES",
    "DOWNREGULATES",
    "ASSOCIATED_WITH",
    "CAUSES",
    "TREATS",
    "PREVENTS",
    "BIOMARKER_OF",
    "PART_OF",
    "SUBTYPE_OF",
    "SAME_AS",
    "EXPRESSED_IN",
    "LOCATED_IN",
    "CO_OCCURS_WITH"
  ]
}
```

Behavior:

- At startup, `BioKG-Mason` read this file and loads these labels/types.
- Loaded labels/types are favored (but not forced) when assigning labels and relationship types to extracted triplets. This keeps the schema consistent while allowing some flexibility per run, without modifying `schema.json` on disk.

---

## 7. Input PDFs

Place all PDFs in the directory specified by `pdf.input_dir`, e.g.:

```text
pdfs/
  12345678.pdf
  87654321.pdf
```

Requirements:

- **Filename (without extension) is the PMID**.  
  Example: `12345678.pdf` → `pmid="12345678"`.

The pipeline will process all `*.pdf` files in `pdf.input_dir`.

---

## 8. Running the Pipeline

### 8.1. Basic run

From the project root:

```bash
pdm run python -m biokg_agents.pipeline
```

This will:

1. Load `config.yml`.
2. Initialize:
   - LangChain chat model (`ChatOpenAI` or `ChatOllama`, `temperature=0.0`).
   - Embeddings client (OpenAI or sentence-transformers).
   - Neo4j client.
   - PDF reader, triplet extractor, entity matcher, validator, schema matcher.
3. Process all PDFs in `pdf.input_dir`.
4. Populate the Neo4j database.

### 8.2. Run with logging to file (recommended for long runs)

```bash
nohup pdm run python -u -m biokg_agents.pipeline > pipeline.log 2>&1 &
tail -f pipeline.log
```

Notes: `tail -f` lets you monitor progress.

---

## 10. Neo4j Data Model

### 10.1. Nodes

- Labels:
  - `PROTEIN`, `GENE`, `RNA`, `DISEASE`, `DRUG`, `PATHWAY`, etc.
- Properties:
  - `entity_id` (string, UUID, used for matching)
  - `name` (string, canonical entity name or surface form)
  - `embedding` (list of floats)

To show an example of a node in Neo4j:

```cypher
MATCH (n)
RETURN labels(n) AS labels, n.entity_id AS entity_id, n.name AS name
LIMIT 10;
```

### 10.2. Relationships

- Types:
  - `ACTIVATES`, `INHIBITS`, `ASSOCIATED_WITH`, `MODULATES`, etc.
- Properties:
  - `pmid` (string)
  - `source_text` (string, paragraph or flattened table text)
  - `source_kind` (`"paragraph"` or `"table"`)
  - `page` (integer, page number in PDF)
  - `table_index` (integer or null; index for tables)

To show an example of a triplet in Neo4j:

```cypher
MATCH (s)-[r]->(o)
RETURN s.name AS subject, type(r) AS rel_type, o.name AS object, r.pmid AS pmid, r.source_kind AS kind
LIMIT 10;
```

---

## 11. Troubleshooting

### 11.1. No relationships in Neo4j

Check:

```cypher
MATCH ()-[r]->() RETURN count(r) AS rel_count;
```

If `rel_count = 0`:

- Ensure `create_relationship` uses `MERGE (s)-[r:TYPE]->(o) SET ...`.
- Verify nodes actually have `entity_id` set.
- Make sure you’re viewing the **same database** as the driver (e.g., `:use neo4j` vs `:use ` in Browser).
- Check `pipeline.log` for `[Neo4j] create_relationship: ... rel_id=...` debug prints if you added them.

### 11.2. Sentence-transformers model errors

If you see errors about `trust_remote_code` or missing `einops`:

- Use a standard model like `sentence-transformers/all-mpnet-base-v2`, or
- Add `einops` via PDM:
  ```bash
  pdm add einops
  ```
- Be cautious with `trust_remote_code=True` for HF models.

### 11.3. Config file not found

If you see `FileNotFoundError: 'config.yml'`, ensure:

- `config.yml` exists in the project root.
- You’re running from the project root.
- The code’s `load_config("config.yml")` matches your filename.

---

## 12. Extending BioKG-Mason

Common extension points:

- **Triplet extraction**:
  - Edit `modules/triplet_extractor.py` to refine prompts or add domain hints.

- **Entity resolution**:
  - Adjust `similarity_threshold` in `config.yml`.
  - Add domain-specific constraints in `EntityMatchValidator` prompts.

- **Schema matching**:
  - Refine label/type suggestions by editing `schema.json`.
  - Adjust `SYSTEM_PROMPT` in `schema_matcher.py` for your subdomain.

- **Provenance**:
  - Add more properties to nodes/relationships (e.g., sentence index, section name).

---


## Citation

If you use this software, please cite it as below.

Title: "BioKG-Mason: Biomedical Knowledge Graph Construction with LLM Agents"

Authors:
  - Family-names: Mohamed

    Given-names: Ahmed Hossameldin
    
    Email: Ahmed.H.H.Mohamed@outlook.com

repository-code: "https://github.com/Ahmed-Mohamed96/BioKG-Mason"

version: "0.1.0"

date-released: "16-12-2025"


## License
MIT

