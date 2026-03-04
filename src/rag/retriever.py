import chromadb
import os
import time
import fitz
from rank_bm25 import BM25Okapi


class VectorStore:
    def __init__(self):
        from chromadb.utils import embedding_functions

        # Use OpenAI embeddings — avoids downloading 400MB local model (OOM on Render 512MB)
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )

        # Use EphemeralClient — Render free tier has no persistent disk, so PersistentClient is useless
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="manufacturing_sops",
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        self.bm25 = None
        self.raw_chunks = []
        self.chunk_sources = {}

    def search(self, query: str, n_results: int = 3):
        """Hybrid search: ChromaDB (semantic) + BM25 (keyword) merged via Reciprocal Rank Fusion."""
        chroma_results = self.collection.query(query_texts=[query], n_results=10)

        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked_docs = [doc for _, doc in sorted(zip(bm25_scores, self.raw_chunks), reverse=True)][:10]

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

    def chunk_text(self, text: str, max_words: int = 100):
        """Sentence-boundary chunking with 1-sentence overlap."""
        sentences = text.split(". ")
        chunks = []
        current_chunk = []
        current_word_count = 0

        for sentence in sentences:
            word_count = len(sentence.split(" "))

            if current_word_count + word_count > max_words and len(current_chunk) > 0:
                chunk_text = ". ".join(current_chunk) + "."
                # HARD LIMIT: truncate to 4000 chars to guarantee < 8192 tokens
                chunks.append(chunk_text[:4000])
                current_chunk = [current_chunk[-1]]
                current_word_count = len(current_chunk[0].split(" "))

            current_chunk.append(sentence)
            current_word_count += word_count

        if current_chunk:
            chunk_text = ". ".join(current_chunk) + "."
            chunks.append(chunk_text[:4000])

        return chunks

    def load_documents_from_folder(self, folder_path: str):
        """Load PDFs and TXT files, chunk them, index in ChromaDB + BM25."""
        filenames = os.listdir(folder_path)
        total_embedded = 0

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

            chunks = self.chunk_text(full_text, max_words=100)
            chunk_ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

            # Batch size = 5 chunks at a time (ultra-safe for OpenAI token limits)
            batch_size = 5
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_ids = chunk_ids[i:i + batch_size]
                try:
                    self.collection.add(documents=batch_chunks, ids=batch_ids)
                    total_embedded += len(batch_chunks)
                except Exception as e:
                    print(f"⚠️ Batch {i}-{i+batch_size} of {filename} failed: {e}")
                    # Try one-by-one as fallback
                    for j, (single_chunk, single_id) in enumerate(zip(batch_chunks, batch_ids)):
                        try:
                            self.collection.add(documents=[single_chunk], ids=[single_id])
                            total_embedded += 1
                        except Exception as inner_e:
                            print(f"  ❌ Skipping chunk {single_id}: {inner_e}")
                # Rate-limit protection: 0.3s pause between API calls
                time.sleep(0.3)

            self.raw_chunks.extend(chunks)

            for chunk in chunks:
                self.chunk_sources[chunk] = filename

            print(f"  ✅ {filename}: {len(chunks)} chunks embedded")

        # Build BM25 keyword index (pure CPU, no API calls)
        tokenized_corpus = [chunk.lower().split(" ") for chunk in self.raw_chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"✅ Total: {total_embedded} chunks indexed from {len(filenames)} files")
        print(f"✅ BM25 keyword index built")


if __name__ == "__main__":
    retriever = VectorStore()
    retriever.load_documents_from_folder("data")