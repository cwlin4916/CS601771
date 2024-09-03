import pickle
import faiss
import numpy as np

# Load the embeddings
evidence_embeddings_pkl = "/Users/miltonlin/Documents/GitHub/SelfSupervisedLearning/scifact_evidence_embeddings.pkl"
claims_embedding_pkl = "/Users/miltonlin/Documents/GitHub/SelfSupervisedLearning/scifact_claim_embeddings.pkl"

with open(evidence_embeddings_pkl, "rb") as f:
    evidence_embeddings = pickle.load(f)

with open(claims_embedding_pkl, "rb") as f:
    claims_embeddings = pickle.load(f)

# Extract and convert embeddings to numpy arrays
evidence_embedding_values = list(evidence_embeddings.values())
evidence_embeddings_np = np.array(evidence_embedding_values, dtype=np.float32)

claims_embedding_values = list(claims_embeddings.values())
claims_embeddings_np = np.array(claims_embedding_values, dtype=np.float32)

# Initialize the FAISS index
index = faiss.IndexFlatIP(evidence_embeddings_np.shape[1])
index.add(evidence_embeddings_np)

# Perform the search for each claim
k = 5
D, I = index.search(claims_embeddings_np, k)

# Example of what D and I look like
print("Example of indices of closest documents (I):", I[0])
print("Example of distances to closest documents (D):", D[0])

# Calculate MRR as an example
def calculate_mrr(I):
    reciprocal_ranks = []
    for indices in I:
        reciprocal_ranks.append(1.0 / (indices[0] + 1))  # Assuming the first document in I is the relevant one
    return np.mean(reciprocal_ranks)

mrr_score = calculate_mrr(I)
print(f"MRR Score: {mrr_score}")
