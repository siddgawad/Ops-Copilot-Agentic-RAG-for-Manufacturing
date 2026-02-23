import chromadb 
import os

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./.chromadb") #currently on ssd for mvp will host ona ws high eprformance clsuter for perfromance ot use ram decated server ovcer ssd for 100x faster results 
        self.collection = self.client.get_or_create_collection(name="manufacturing_sops",metadata={"hnsw:space":"cosine"})

    def add_documents(self,documents:list[str],ids:list[str]):
        self.collection.add(documents=documents,ids=ids)
    
    def search(self, query:str, n_results:int=3):
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return results

    def chunk_text(self, text:str, max_words: int=50, overlap: int=10):
        #split the text into sentences using the period 
        sentences = text.split(". ")
        chunks = []
        current_chunk=[]
        current_word_count=0

        for sentence in sentences:
            word_count = len(sentence.split(" "))

            if(current_word_count + word_count > max_words and len(current_chunk) > 0):

                chunks.append(". ".join(current_chunk)+".") # explaijn this line 

                current_chunk = [current_chunk[-1]]
                current_word_count = len(current_chunk[0].split(" "))

            current_chunk.append(sentence)
            current_word_count+=word_count

        if current_chunk:
            chunks.append(". ".join(current_chunk)+".")

        return chunks 


    def load_documents_from_folder(self, folder_path:str):
        filenames = os.listdir(folder_path)
        for filename in filenames:
            fileapth = os.path.join(folder_path,filename)
            with open(fileapth,"r") as f:
                text = f.read()
                
                #chop text into pieces using our new method 
                chunks = self.chunk_text(text,max_words=50)

                #we need unique IDs for every single chunk 
                chunk_ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]

                #add the chunks to chromaDB instead of full text 
                self.collection.add(documents=chunks, ids=chunk_ids)




if __name__ == "__main__":
    from generator import generate_answer

    # 1. Turn on the machine
    db = VectorStore()

    db.load_documents_from_folder("data")
    print("All SOPs loaded!")

    # 2. Ask a question
    question = "What vibration level requires immediate machine shutdown?"
    print(f"\nSearching for: '{question}'")
    results = db.search(query=question, n_results=3)

    # 3. Show the raw retrieved chunks
    print("\n--- RETRIEVED CHUNKS ---")
    for doc, distance in zip(results['documents'][0], results['distances'][0]):
        print(f"  Score: {1 - distance:.3f} | {doc[:100]}...")

    # 4. Send chunks + question to OpenAI for a real answer
    print("\n--- AI GENERATED ANSWER ---")
    answer = generate_answer(question, results['documents'][0])
    print(answer)