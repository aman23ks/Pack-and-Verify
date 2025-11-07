# Pack-and-Verify (Quick POC)

Free-tier POC: Unstructured (hi_res) to segment PDFs (text/tables/images),
Gemini for vision+QA+embeddings, Pinecone for vector search, and a budgeted
packer (gain-per-token + MMR).

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # paste your keys
mkdir -p data/papers   # drop 3–5 PDFs here
python cli.py ingest data/papers
python cli.py ask "What does Table 2 report?"
