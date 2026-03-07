import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class RAGAgent(ABC):
    """Base class for RAG functionality for language/library documentation"""

    @abstractmethod
    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documentation chunks

        Args:
            query: Query string
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant documentation chunks
        """

    @abstractmethod
    async def get_context(self, query: str, language: str = None) -> str:
        """
        Get formatted documentation context

        Args:
            query: Query describing what documentation is needed
            language: Programming language filter (e.g., "C", "Python")

        Returns:
            Formatted documentation context string to add to prompt
        """

    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add new documentation to the RAG store

        Args:
            documents: List of documents with 'content', 'source', 'language' fields
        """


class NoOpRAGAgent(RAGAgent):
    """Empty implementation - does nothing"""

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return []

    async def get_context(self, query: str, language: str = None) -> str:
        return ""

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        pass


class ChromaRAGAgent(RAGAgent):
    """
    RAG agent using in-memory ChromaDB for vector storage.
    
    Loads markdown documents from a specified directory, chunks them,
    creates embeddings using an OpenAI-compatible API, and enables
    semantic search for retrieval.
    """

    DEFAULT_CHUNK_SIZE = 500
    DEFAULT_CHUNK_OVERLAP = 50

    def __init__(
        self,
        data_dir: str,
        embedding_base_url: str = "http://localhost:8000/v1",
        embedding_api_key: str = None,
        embedding_model: str = "bge-m3",
        chunk_size: int = None,
        chunk_overlap: int = None,
        collection_name: str = "documents",
    ):
        """
        Initialize the Chroma RAG agent.

        Args:
            data_dir: Directory containing markdown documents to index
            embedding_base_url: Base URL for OpenAI-compatible embedding API
            embedding_api_key: API key for embedding API
            embedding_model: Model name for embeddings
            chunk_size: Size of text chunks (default: 500 chars)
            chunk_overlap: Overlap between chunks (default: 50 chars)
            collection_name: Name for the Chroma collection
        """
        self.data_dir = Path(data_dir)
        self.embedding_base_url = embedding_base_url
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size or self.DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or self.DEFAULT_CHUNK_OVERLAP
        self.collection_name = collection_name
        
        self._client = None
        self._collection = None
        self._initialized = False
        self._doc_count = 0

    def _get_embedding_client(self):
        """Get OpenAI-compatible embedding client."""
        import openai
        return openai.Client(
            base_url=self.embedding_base_url,
            api_key=self.embedding_api_key or "dummy-key"
        )

    async def _ensure_initialized(self):
        """Initialize ChromaDB and index documents if not already done."""
        if self._initialized:
            return

        try:
            import chromadb
        except ImportError:
            logger.error("ChromaDB not installed. Install with: pip install chromadb")
            self._initialized = True
            return

        # Create in-memory ChromaDB client
        self._client = chromadb.Client(chromadb.config.Settings(
            is_persistent=False,
            anonymized_telemetry=False,
        ))

        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # Load and index documents
        await self._load_and_index_documents()
        self._initialized = True
        
        logger.info(f"ChromaRAGAgent initialized with {self._doc_count} documents indexed")

    def _generate_doc_id(self, content: str, source: str) -> str:
        """Generate a unique ID for a document chunk."""
        text = f"{source}:{content[:100]}"
        return hashlib.md5(text.encode()).hexdigest()

    def _chunk_text(self, text: str, source: str) -> List[Dict[str, str]]:
        """
        Split text into overlapping chunks.

        Returns:
            List of dicts with 'id', 'content', 'source' keys
        """
        chunks = []
        
        if len(text) <= self.chunk_size:
            chunks.append({
                "id": self._generate_doc_id(text, source),
                "content": text,
                "source": source,
            })
            return chunks

        # Simple character-based chunking with overlap
        start = 0
        chunk_idx = 0
        while start < len(text):
            end = start + self.chunk_size
            
            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for sentence boundary
                for sep in ["\n\n", "\n", ". ", "! ", "? "]:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.chunk_size // 2:
                        end = start + last_sep + len(sep)
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append({
                    "id": self._generate_doc_id(chunk_text, f"{source}:{chunk_idx}"),
                    "content": chunk_text,
                    "source": source,
                })
                chunk_idx += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI-compatible API."""
        client = self._get_embedding_client()
        
        response = client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        
        # Sort by index to maintain order
        sorted_embeddings = sorted(response.data, key=lambda x: x.index)
        return [emb.embedding for emb in sorted_embeddings]

    async def _load_and_index_documents(self):
        """Load markdown files from data_dir and index them."""
        if not self.data_dir.exists():
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return

        markdown_files = list(self.data_dir.glob("*.md")) + \
                        list(self.data_dir.glob("*.markdown"))
        
        if not markdown_files:
            logger.warning(f"No markdown files found in {self.data_dir}")
            return

        all_chunks = []
        all_ids = []
        all_metadatas = []

        for file_path in markdown_files:
            logger.info(f"Loading document: {file_path.name}")
            
            try:
                content = file_path.read_text(encoding="utf-8")
            except Exception as e:
                logger.error(f"Failed to read {file_path.name}: {e}")
                continue

            # Extract language from filename or content
            language = self._detect_language(file_path.name, content)
            
            # Chunk the document
            chunks = self._chunk_text(content, file_path.name)
            
            for chunk in chunks:
                all_chunks.append(chunk["content"])
                all_ids.append(chunk["id"])
                all_metadatas.append({
                    "source": file_path.name,
                    "language": language,
                    "chunk_id": chunk["id"],
                })

        if not all_chunks:
            logger.warning("No chunks created from documents")
            return

        # Create embeddings in batches
        batch_size = 32
        for i in range(0, len(all_chunks), batch_size):
            batch_texts = all_chunks[i:i + batch_size]
            batch_ids = all_ids[i:i + batch_size]
            batch_metadatas = all_metadatas[i:i + batch_size]
            
            try:
                embeddings = self._create_embeddings(batch_texts)
                
                self._collection.add(
                    embeddings=embeddings,
                    ids=batch_ids,
                    metadatas=batch_metadatas,
                    documents=batch_texts,
                )
                self._doc_count += len(batch_texts)
                
            except Exception as e:
                logger.error(f"Failed to create embeddings for batch {i}: {e}")

    def _detect_language(self, filename: str, content: str) -> str:
        """Detect programming language from filename or content."""
        filename_lower = filename.lower()
        
        # Check filename patterns
        if "python" in filename_lower or filename.endswith(".py"):
            return "Python"
        elif "c" in filename_lower or filename.endswith(".c") or filename.endswith(".h"):
            return "C"
        elif "cpp" in filename_lower or filename.endswith(".cpp") or filename.endswith(".hpp"):
            return "C++"
        elif "java" in filename_lower or filename.endswith(".java"):
            return "Java"
        elif "js" in filename_lower or filename.endswith(".js"):
            return "JavaScript"
        elif "rust" in filename_lower or filename.endswith(".rs"):
            return "Rust"
        elif "go" in filename_lower or filename.endswith(".go"):
            return "Go"
        
        # Check content for language hints
        if "#include" in content and ("printf" in content or "malloc" in content):
            return "C"
        elif "import " in content and ("print(" in content or "def " in content):
            return "Python"
        
        return "unknown"

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documentation chunks.

        Args:
            query: Query string
            top_k: Number of chunks to retrieve

        Returns:
            List of relevant documentation chunks with metadata
        """
        await self._ensure_initialized()
        
        if not self._collection:
            logger.warning("ChromaDB not initialized, returning empty results")
            return []

        try:
            # Create embedding for query
            query_embedding = self._create_embeddings([query])[0]
            
            # Query the collection
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            
            # Format results
            retrieved = []
            if results and results["documents"]:
                for i, doc in enumerate(results["documents"][0]):
                    retrieved.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else None,
                    })
            
            logger.debug(f"Retrieved {len(retrieved)} documents for query: {query[:50]}...")
            return retrieved
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []

    async def get_context(self, query: str, language: str = None) -> str:
        """
        Get formatted documentation context.

        Args:
            query: Query describing what documentation is needed
            language: Programming language filter (e.g., "C", "Python")

        Returns:
            Formatted documentation context string to add to prompt
        """
        results = await self.retrieve(query, top_k=5)
        
        if not results:
            return ""

        # Filter by language if specified
        if language:
            results = [
                r for r in results 
                if r.get("metadata", {}).get("language", "").lower() == language.lower()
                or r.get("metadata", {}).get("language") == "unknown"
            ]

        if not results:
            return ""

        # Format context
        context_parts = []
        for i, result in enumerate(results, 1):
            source = result.get("metadata", {}).get("source", "unknown")
            content = result["content"]
            distance = result.get("distance", 0)
            
            context_parts.append(
                f"[Document {i} - Source: {source} - Relevance: {1 - distance:.2f}]\n"
                f"{content}\n"
            )

        context = "\n---\n\n".join(context_parts)
        
        logger.debug(f"Generated context with {len(context_parts)} documents")
        return context

    async def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add new documentation to the RAG store.

        Args:
            documents: List of documents with 'content', 'source', 'language' fields
        """
        await self._ensure_initialized()
        
        if not self._collection:
            logger.warning("ChromaDB not initialized, cannot add documents")
            return

        all_chunks = []
        all_ids = []
        all_metadatas = []

        for doc in documents:
            content = doc.get("content", "")
            source = doc.get("source", "unknown")
            language = doc.get("language", "unknown")
            
            chunks = self._chunk_text(content, source)
            
            for chunk in chunks:
                all_chunks.append(chunk["content"])
                all_ids.append(chunk["id"])
                all_metadatas.append({
                    "source": source,
                    "language": language,
                    "chunk_id": chunk["id"],
                })

        if not all_chunks:
            logger.warning("No chunks created from documents")
            return

        # Create embeddings and add to collection
        try:
            embeddings = self._create_embeddings(all_chunks)
            
            self._collection.add(
                embeddings=embeddings,
                ids=all_ids,
                metadatas=all_metadatas,
                documents=all_chunks,
            )
            self._doc_count += len(all_chunks)
            
            logger.info(f"Added {len(all_chunks)} chunks from {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG index."""
        return {
            "initialized": self._initialized,
            "document_count": self._doc_count,
            "data_dir": str(self.data_dir),
            "embedding_model": self.embedding_model,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
        }
