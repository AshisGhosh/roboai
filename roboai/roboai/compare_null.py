import json
import os
import numpy as np
import matplotlib.pyplot as plt
from fastembed import TextEmbedding

# Cosine similarity function using numpy
def cosine_similarity_np(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return float(dot_product / (norm_vec1 * norm_vec2))

# Function to load data from a file
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Function to compare datasets and compute cosine similarities
def calculate_semantic_similarity_per_run(synthetic_data, gt_data, embed_model):
    """
    Compare each entry in the synthetic data directly to the corresponding index entry in the ground truth data.
    Output the results as JSON and create a histogram of the cosine similarities.
    """
    results = []
    all_cosine_similarities = []

    synthetic_keys = sorted(synthetic_data['synthetic_responses'].keys(), key=lambda x: int(x))
    gt_keys = sorted(gt_data['validation_data'].keys())

    if len(synthetic_keys) != len(gt_keys):
        print("Warning: The number of entries in ground truth and synthetic datasets does not match.")
        return {}, []

    best_match_scores = []

    for syn_key, gt_key in zip(synthetic_keys, gt_keys):
        syn_entry = synthetic_data['synthetic_responses'][syn_key]
        gt_entry = gt_data['validation_data'][gt_key]

        syn_names = syn_entry['objects']['names']
        gt_names = gt_entry['objects']['names']

        syn_embeddings = list(embed_model.embed(syn_names))
        gt_embeddings = list(embed_model.embed(gt_names))

        run_result = {
            "synthetic_key": syn_key,
            "ground_truth_key": gt_key,
            "comparisons": []
        }

        for syn_idx, syn_embed in enumerate(syn_embeddings):
            similarities = [cosine_similarity_np(syn_embed, gt_embed) for gt_embed in gt_embeddings]
            all_cosine_similarities.extend(similarities)
            run_result["comparisons"].append({
                "synthetic_name": syn_names[syn_idx],
                "similarities": similarities,
                "max_similarity": max(similarities),
                "average_similarity": np.mean(similarities)
            })

        results.append(run_result)
        best_match_scores.append(max(similarities))

    mean_best_match = np.mean(best_match_scores)
    std_best_match = np.std(best_match_scores)

    mean_similarity = np.mean(all_cosine_similarities)
    std_similarities = np.std(all_cosine_similarities)
    
    results_summary = {
    "results": results,
    "mean_similarity": mean_similarity,
    "std_deviation": std_similarities,
    "mean_best_match": mean_best_match,
    "std_best_match": std_best_match
    }

    return results_summary, all_cosine_similarities, best_match_scores

# Paths to your JSON files
gt_path = "/app/shared/data/image_exports/20240505_210115/analysis/20240506_050006_llava-13b/output.json"
synthetic_path = "/app/shared/data/image_exports/20240505_210115/llava_synthetic_responses.json"

# Load the embedding model
embed_model = TextEmbedding("mixedbread-ai/mxbai-embed-large-v1")

# Load the data
gt_data = load_data(gt_path)
synthetic_data = load_data(synthetic_path)

# Assuming the function call has been updated accordingly
results_summary, all_cosine_similarities, best_match_scores = calculate_semantic_similarity_per_run(synthetic_data, gt_data, embed_model)

# Generate JSON output
json_output = json.dumps(results_summary, indent=4)
filename = '/app/shared/data/null_stats.json'
with open(filename, 'w') as f:
    f.write(json_output)
    print(f"Results saved to {filename}")

# Generate histogram
# Generate histogram for best match scores
plt.hist(best_match_scores, bins=30, color='blue', alpha=0.7)
plt.title('Distribution of Best Match Cosine Similarities - Null vs GT')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
n = len(best_match_scores)
plt.text(0.95, 0.95, f'n = {n}', ha='right', va='top', transform=plt.gca().transAxes)
plt.grid(True)
plt.savefig('/app/shared/data/null_histogram.png')
plt.show()
print("Best match histogram saved to /app/shared/data/null_histogram.png")

# Change ownership to match the host user, if the script runs as root
if os.getuid() == 0:
    host_uid = 1000  # Replace with actual host user ID
    host_gid = 1000  # Replace with actual host group ID
    os.chown("/app/shared/data/null_histogram.png", host_uid, host_gid)
    os.chown("/app/shared/data/null_stats.json", host_uid, host_gid)