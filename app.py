import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

# =========================
# Local Embedding Function
# =========================
class LocalEmbeddingFunction:

    def __init__(self):
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def __call__(self, input):
        return self.model.encode(input).tolist()

    def name(self):
        return "local-embedding-function"


# =========================
# Initialize ChromaDB
# =========================
client = chromadb.PersistentClient(
    path="./chroma_db"
)

embedding_function = LocalEmbeddingFunction()

collection = client.get_or_create_collection(
    name="tech_docs",
    embedding_function=embedding_function
)

# =========================
# Sample Documents
# =========================
documents = [
    "ChromaDB supports metadata filtering and CRUD operations",
    "Sentence Transformers generate semantic embeddings for search applications",
    "HNSW indexing enables efficient approximate nearest neighbor search",
    "Retrieval evaluation requires precision@k and recall@k metrics",
    "Vector databases store embeddings for similarity search"
]

metadata = [
    {"category": "chromadb", "source": "docs"},
    {"category": "embeddings", "source": "research"},
    {"category": "indexing", "source": "paper"},
    {"category": "evaluation", "source": "tutorial"},
    {"category": "vector-db", "source": "textbook"}
]

ids = [f"doc{i+1}" for i in range(len(documents))]

# =========================
# Insert Documents
# =========================
collection.upsert(
    documents=documents,
    metadatas=metadata,
    ids=ids
)

print(f"\nAdded {len(ids)} documents to ChromaDB")

# =========================
# Semantic Search Pipeline
# =========================
def search_pipeline(
    query,
    n_results=3,
    filters=None
):

    query_embedding = embedding_function([query])[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=filters
    )

    return pd.DataFrame({
        "Document": results["documents"][0],
        "ID": results["ids"][0],
        "Distance": results["distances"][0],
        "Metadata": results["metadatas"][0]
    })


# =========================
# Example Search
# =========================
print("\n========== SEMANTIC SEARCH ==========\n")

search_results = search_pipeline(
    "How to measure search quality?"
)

print(search_results)

# =========================
# Ground Truth Queries
# =========================
test_queries = {
    "q1": {
        "text": "ChromaDB features",
        "relevant_ids": ["doc1"]
    },
    "q2": {
        "text": "Similarity measurement techniques",
        "relevant_ids": ["doc5"]
    },
    "q3": {
        "text": "Evaluation metrics for search",
        "relevant_ids": ["doc4"]
    }
}

# =========================
# Retrieval Evaluation
# =========================
def evaluate_retrieval(k=2):

    results = []

    for _, data in test_queries.items():

        query_embedding = embedding_function(
            [data["text"]]
        )[0]

        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )

        retrieved_ids = search_results["ids"][0]

        relevant_set = set(data["relevant_ids"])
        retrieved_set = set(retrieved_ids)

        true_positives = len(
            relevant_set & retrieved_set
        )

        precision = true_positives / k

        recall = (
            true_positives / len(relevant_set)
            if relevant_set else 0
        )

        reciprocal_rank = 0

        for rank, doc_id in enumerate(
            retrieved_ids,
            1
        ):

            if doc_id in relevant_set:
                reciprocal_rank = 1 / rank
                break

        results.append({
            "Query": data["text"],
            "Precision@K": precision,
            "Recall@K": recall,
            "MRR": reciprocal_rank
        })

    return pd.DataFrame(results)


# =========================
# Run Evaluation
# =========================
print("\n========== RETRIEVAL EVALUATION ==========\n")

eval_results = evaluate_retrieval(k=2)

print(eval_results)

# =========================
# Average Metrics
# =========================
avg_metrics = pd.DataFrame({
    "Metric": [
        "Precision@2",
        "Recall@2",
        "MRR"
    ],
    "Average": [
        eval_results["Precision@K"].mean(),
        eval_results["Recall@K"].mean(),
        eval_results["MRR"].mean()
    ]
})

print("\n========== AVERAGE METRICS ==========\n")

print(avg_metrics)