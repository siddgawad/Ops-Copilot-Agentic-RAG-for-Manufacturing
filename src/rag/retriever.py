import chromadb 
import os
import fitz
from rank_bm25 import BM25Okapi

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./.chromadb")
        self.collection = self.client.get_or_create_collection(
            name="manufacturing_sops",
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25 = None
        self.raw_chunks = []
        self.chunk_sources = {}  # Maps chunk text -> source filename

    def add_documents(self, documents: list[str], ids: list[str]):
        self.collection.add(documents=documents, ids=ids)
    
    def search(self, query: str, n_results: int = 3):
        """Hybrid search: ChromaDB (semantic) + BM25 (keyword) merged via Reciprocal Rank Fusion."""
        # 1. Semantic search via ChromaDB
        chroma_results = self.collection.query(query_texts=[query], n_results=10)

        # 2. Keyword search via BM25
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked_docs = [doc for _, doc in sorted(zip(bm25_scores, self.raw_chunks), reverse=True)][:10]

        # 3. Reciprocal Rank Fusion (k=60)
        fused_scores = {}
        
        chroma_docs = chroma_results['documents'][0]
        for rank, doc in enumerate(chroma_docs):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            fused_scores[doc] += 1 / (rank + 60)
            
        for rank, doc in enumerate(bm25_ranked_docs):
            if doc not in fused_scores:
                fused_scores[doc] = 0
            fused_scores[doc] += 1 / (rank + 60)
            
        # 4. Sort by fused score and return top N with source metadata
        sorted_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        results = []
        for doc, score in sorted_docs[:n_results]:
            source = self.chunk_sources.get(doc, "Unknown")
            results.append({
                "text": doc,
                "source": source,
                "score": round(score, 6)
            })
        return results

    def chunk_text(self, text: str, max_words: int = 50, overlap: int = 50):
        """Sentence-boundary chunking with 1-sentence overlap to prevent context severing."""
        sentences = text.split(". ")
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split(" "))

            if current_word_count + word_count > max_words and len(current_chunk) > 0:
                chunks.append(". ".join(current_chunk) + ".")
                current_chunk = [current_chunk[-1]]  # overlap: keep last sentence
                current_word_count = len(current_chunk[0].split(" "))

            current_chunk.append(sentence)
            current_word_count += word_count

        if current_chunk:
            chunks.append(". ".join(current_chunk) + ".")

        return chunks 

    def load_documents_from_folder(self, folder_path: str):
        """Load PDFs and TXT files, chunk them, index in ChromaDB + BM25."""
        filenames = os.listdir(folder_path)
        for filename in filenames:
            filepath = os.path.join(folder_path, filename)
            full_text = ""

            if filename.endswith(".pdf"):
                doc = fitz.open(filepath)
                for page in doc:
                    full_text += page.get_text("text")
            elif filename.endswith(".txt"):
                with open(filepath, encoding="utf-8") as f:
                    full_text = f.read()
            else:
                continue 

            if len(full_text) == 0:
                continue

            chunks = self.chunk_text(full_text, max_words=350)
            chunk_ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

            self.collection.add(documents=chunks, ids=chunk_ids)
            self.raw_chunks.extend(chunks)

            # Track source filename for each chunk (for citations)
            for chunk in chunks:
                self.chunk_sources[chunk] = filename

        # Build BM25 keyword index
        tokenized_corpus = [chunk.lower().split(" ") for chunk in self.raw_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ Indexed {len(self.raw_chunks)} chunks from {len(filenames)} files")
        print(f"✅ BM25 keyword index built")


if __name__ == "__main__":
    retriever = VectorStore()
    retriever.load_documents_from_folder("data")