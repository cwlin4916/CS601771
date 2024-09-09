# Import required libraries
import pickle
import faiss
import numpy as np
from datasets import load_dataset

# Step 1: Load the SciFact claims dataset
# The SciFact dataset contains claims in train, validation, and test splits
claims = load_dataset("allenai/scifact", "claims", trust_remote_code=True)

# Extract claims from the train, validation, and test sets
claims_train = claims['train']
claims_validation = claims['validation']
claims_test = claims['test']

# Collect all claim IDs across the three splits
train_claim_ids = [claim['id'] for claim in claims_train]
validation_claim_ids = [claim['id'] for claim in claims_validation]
test_claim_ids = [claim['id'] for claim in claims_test]
all_claim_ids = train_claim_ids + validation_claim_ids + test_claim_ids

print(f"Total Claims in All Splits: {len(all_claim_ids)}")

# Step 2: Load embeddings for the evidence and claims
# These files contain the pre-computed embeddings for the evidence and claims
evidence_embeddings_pkl = "/Users/miltonlin/Documents/GitHub/SelfSupervisedLearning/scifact_evidence_embeddings.pkl"
claims_embeddings_pkl = "/Users/miltonlin/Documents/GitHub/SelfSupervisedLearning/scifact_claim_embeddings.pkl"

with open(evidence_embeddings_pkl, "rb") as f:
    evidence_embeddings = pickle.load(f)

with open(claims_embeddings_pkl, "rb") as f:
    claims_embeddings = pickle.load(f)

# Convert the embeddings to NumPy arrays for FAISS
evidence_embedding_values = list(evidence_embeddings.values())
evidence_embeddings_np = np.array(evidence_embedding_values, dtype=np.float32)

claims_embedding_values = list(claims_embeddings.values())
claims_embeddings_np = np.array(claims_embedding_values, dtype=np.float32)

print(f"Loaded {len(evidence_embeddings)} evidence embeddings.")
print(f"Loaded {len(claims_embeddings)} claims embeddings.")

# Step 3: Filter claims that have matching embeddings
# Extract the claim IDs from the claims embeddings to ensure they match the loaded dataset
embedding_claim_ids = [claim[0] for claim in claims_embeddings.keys()]
matching_claims = set(all_claim_ids).intersection(set(embedding_claim_ids))

print(f"Matching Claims: {len(matching_claims)}")

# Filter claims to retain only those that have matching embeddings
filtered_claims = [claim for claim in claims_train if claim['id'] in matching_claims]
print(f"Total Filtered Claims: {len(filtered_claims)}")

# Step 4: Extract gold labels for the filtered claims
# This will serve as the ground truth for evaluating the retrieval system (using `evidence_doc_id`)
gold_labels = {}
for claim in filtered_claims:
    claim_id = claim['id']
    evidence_doc_id = claim.get('evidence_doc_id', None)
    evidence_label = claim.get('evidence_label', None)
    if evidence_doc_id and evidence_label:
        gold_labels[claim_id] = {'evidence_doc_id': evidence_doc_id, 'evidence_label': evidence_label}

if gold_labels:
    example_claim_id = list(gold_labels.keys())[0]
    print(f"Gold label for claim {example_claim_id}: {gold_labels[example_claim_id]}")

# Step 5: Build FAISS Index
# FAISS is used to efficiently search the evidence embeddings based on the query claim embeddings
d = evidence_embeddings_np.shape[1]  # Dimension of the embeddings (e.g., 768 for BERT embeddings)
index = faiss.IndexFlatL2(d)  # Using L2 distance for nearest neighbor search
index.add(evidence_embeddings_np)  # Add the evidence embeddings to the FAISS index

print(f"FAISS index contains {index.ntotal} vectors.")

# Step 6: Perform FAISS search
# Retrieve the k-nearest neighbors for each claim embedding
total_evidence_docs = evidence_embeddings_np.shape[0]  # Total number of evidence documents
# k=total_evidence_docs # Retrieve all evidence documents for each claim 
k=3
distances, indices = index.search(claims_embeddings_np, k)

# Step 7: Map FAISS indices to actual document IDs
# The FAISS indices must be mapped to the corresponding document IDs
evidence_doc_ids = list(evidence_embeddings.keys())
retrieved_doc_ids = [[evidence_doc_ids[idx][0] for idx in retrieved_indices] for retrieved_indices in indices]

# Debugging: Print some retrieved document IDs to verify
# for i in range(5):
#     print(f"Claim {i} retrieved doc_ids: {retrieved_doc_ids[i]}")

# Step 8: Evaluate using Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR)
# This function computes MAP and MRR to evaluate the quality of the retrieved evidence
# This function computes MAP and MRR to evaluate the quality of the retrieved evidence
def compute_map_mrr(retrieved_doc_ids, gold_labels):
    avg_precisions = []
    reciprocal_ranks = []

    for i, retrieved_docs in enumerate(retrieved_doc_ids):
        claim_id = embedding_claim_ids[i]  # Get the claim ID
        true_evidence = gold_labels.get(claim_id, {}).get('evidence_doc_id')

        if not true_evidence:
            continue

        relevant_found = False
        precision_at_ranks = []
        relevant_count = 0  # Keep track of relevant docs found
        
        for rank, doc_id in enumerate(retrieved_docs):
            if str(doc_id) == str(true_evidence):  # Check if retrieved doc is the relevant one
                relevant_count += 1
                precision_at_ranks.append(relevant_count / (rank + 1))  # Precision at this rank
                if not relevant_found:
                    reciprocal_ranks.append(1 / (rank + 1))  # Add to MRR if this is the first relevant doc
                    relevant_found = True

        # Average precision is the mean of the precisions at each relevant rank
        if precision_at_ranks:
            avg_precisions.append(sum(precision_at_ranks) / len(precision_at_ranks))
        else:
            avg_precisions.append(0)

    # MAP is the mean of average precision scores
    MAP = sum(avg_precisions) / len(avg_precisions) if avg_precisions else 0
    # MRR is the mean of reciprocal ranks
    MRR = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    return MAP, MRR

# Compute MAP and MRR to evaluate retrieval performance
MAP, MRR = compute_map_mrr(retrieved_doc_ids, gold_labels)

# Print the final evaluation results
print(f"Mean Average Precision (MAP): {MAP}")
print(f"Mean Reciprocal Rank (MRR): {MRR}")
