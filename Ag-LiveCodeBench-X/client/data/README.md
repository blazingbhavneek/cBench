# RAG Data Directory

Place your markdown documentation files here.

When you enable RAG with `--use-rag`, the system will:
1. Load all `.md` and `.markdown` files from this directory
2. Chunk them and create embeddings
3. Store in an in-memory ChromaDB vector database
4. Retrieve relevant context during LLM generation

See the main [README.md](../../README.md#rag-retrieval-augmented-generation) for setup and usage instructions.
