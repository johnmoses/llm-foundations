from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM


class MilvusRAG:
    def __init__(
        self,
        db_path="milvus_rag_db.db",
        collection_name="rag_collection",
        embedding_dim=768,
    ):
        self.client = MilvusClient(db_path)
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        if not self.client.has_collection(collection_name):
            self.client.create_collection(collection_name, dimension=embedding_dim)
        self.document_store = {}

    def insert_documents(self, documents):
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedder.encode(documents, convert_to_numpy=True).tolist()
        res = self.client.insert(self.collection_name, embeddings)
        ids = res["ids"]
        for doc_id, doc_text in zip(ids, documents):
            self.document_store[doc_id] = doc_text
        self.client.flush(self.collection_name)

    def search(self, query, top_k=3):
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
        query_embedding = embedder.encode([query], convert_to_numpy=True).tolist()
        results = self.client.search(self.collection_name, query_embedding, limit=top_k)
        hits = results[0]
        return [self.document_store.get(hit["id"], "") for hit in hits]


def rag_generate(query, rag_client, llm_model, tokenizer, top_k=3, max_length=200):
    retrieved_docs = rag_client.search(query, top_k=top_k)
    context = "\n".join(retrieved_docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = llm_model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
