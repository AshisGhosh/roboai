import argparse
import datetime as dt
import json
import os
import numpy as np
from scipy.stats import kendalltau
from fastembed import TextEmbedding

def load_json_file(filepath):
    """Load JSON data from a file."""
    with open(filepath, 'r') as file:
        return json.load(file)

def parse_responses(response_string):
    """Parse response strings by splitting on commas and removing 'and'."""
    items = response_string.replace(' and', '').split(', ')
    return [item.strip() for item in items if item]

def cosine_similarity_np(vec1, vec2):
    """Compute cosine similarity between two vectors using numpy."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_semantic_similarity(response_names, gt_names, embed_model, similarity_threshold=0.50):
    """
    For each response name, find the ground truth name with the highest cosine similarity.
    Any response names that do not have a matching ground truth >0.50 are reported as unpaired,
    usually when len(response_names) > len(gt_names).
    """
    sem_score_matches = []
    unpaired_responses = []
    all_sem_scores = []
    all_cosine_similarities = []

    response_embeddings = list(embed_model.embed(response_names))
    gt_embeddings = list(embed_model.embed(gt_names))
    all_pairs = []

    for response_idx, (response_name, response_embed) in enumerate(zip(response_names, response_embeddings)):
        similarities = [cosine_similarity_np(response_embed, gt_embed) for gt_embed in gt_embeddings]
        
        similarity_info = {
            "response_name": response_name,
            "response_idx": response_idx,
            "similarities": [{"gt_name": gt_name, "similarity": sim, "gt_idx": gt_idx}
                             for gt_idx, (gt_name, sim) in enumerate(zip(gt_names, similarities))]
        }
        all_sem_scores.append(similarity_info)
        all_cosine_similarities.extend(similarities)  # Collect similarities for statistics

        for gt_idx, sim in enumerate(similarities):
            if sim >= similarity_threshold:
                all_pairs.append({
                    "response_name": response_name,
                    "response_idx": response_idx,
                    "gt_name": gt_names[gt_idx],
                    "gt_name_index": gt_idx,
                    "cosine_similarity": sim
                })

    # Sort all pairs by similarity in descending order
    all_pairs.sort(key=lambda x: x["cosine_similarity"], reverse=True)

    matched_gt_indices = set()
    matched_response_indices = set()

    for pair in all_pairs:
        if pair["gt_name_index"] not in matched_gt_indices and pair["response_idx"] not in matched_response_indices:
            matched_gt_indices.add(pair["gt_name_index"])
            matched_response_indices.add(pair["response_idx"])
            sem_score_matches.append(pair)

    # Find unmatched responses
    unpaired_responses = [{"response_name": name, "response_idx": idx}
                          for idx, name in enumerate(response_names) if idx not in matched_response_indices]

    # Calculate statistics for cosine similarities
    all_sem_score_stats = {
        "mean": float(np.mean(all_cosine_similarities)),
        "std": float(np.std(all_cosine_similarities)),
        "min": float(np.min(all_cosine_similarities)),
        "max": float(np.max(all_cosine_similarities)),
        "count": len(all_cosine_similarities)
    }

    # Calculate statistics for sem_score_matches
    sem_score_match_stats = {
        "mean": float(np.mean([pair['cosine_similarity'] for pair in sem_score_matches])),
        "std": float(np.std([pair['cosine_similarity'] for pair in sem_score_matches])),
        "min": float(min([pair['cosine_similarity'] for pair in sem_score_matches], default=0)),
        "max": float(max([pair['cosine_similarity'] for pair in sem_score_matches], default=0)),
        "count": len(sem_score_matches)
    }
    
    return sem_score_matches, unpaired_responses, sem_score_match_stats, all_sem_scores, all_sem_score_stats

def calculate_count_accuracy(response_count, gt_count):
    # Calculate the count score based on the difference between the response and ground truth counts
    if gt_count != 0:
        count_score = 1 - (abs(response_count - gt_count) / gt_count)
    else:
        count_score = 0 if response_count != 0 else 1  # Both are zero

    return max(0, count_score)

def kendall_tau_normalized(sem_score_matches):
    """
    Calculates normalized Kendall's tau correlation between the ground truth indices
    and the response indices based on semantically matched objects.
    """
    matched_indices = [score["gt_name_index"] for score in sem_score_matches]
    response_indices = [score["response_idx"] for score in sem_score_matches]
    
    # Calculate Kendall's tau
    tau, p_value = kendalltau(matched_indices, response_indices)
    
    # Normalize the Kendall's tau score to be in the range [0, 1]
    normalized_tau = (tau + 1) / 2

    return {
        "normalized_kendall_tau": float(normalized_tau),
        "p_value": float(p_value),
        "matched_indices": matched_indices,
        "response_indices": response_indices
    }

def evaluate_outputs(response_names, gt_names, embed_model):
    """Evaluate outputs using provided complex functions."""
    sem_score_matches, unpaired_responses, sem_score_match_stats, all_sem_scores, all_sem_score_stats = calculate_semantic_similarity(response_names, gt_names, embed_model)
    order_accuracy_result = kendall_tau_normalized(sem_score_matches)
    count_accuracy_score = calculate_count_accuracy(len(response_names), len(gt_names))
    
    final_score = (order_accuracy_result["normalized_kendall_tau"] * count_accuracy_score * sem_score_match_stats["mean"])
    
    return {
        "count_accuracy": count_accuracy_score,
        "order_accuracy": order_accuracy_result,
        "sem_score_matches": sem_score_matches,
        "unpaired_responses": unpaired_responses,
        "sem_score_match_stats": sem_score_match_stats,
        "all_sem_scores": all_sem_scores,
        "all_sem_score_stats": all_sem_score_stats,
        "final_score": final_score
    }
    
def compute_aggregate_stats(scores):
    if scores:
        # Collect scores for each metric
        order_accuracy_scores = []
        average_semantic_scores = []
        count_accuracy_scores = []
        overall_performance_scores = []

        for filename, score_data in scores.items():
            order_accuracy_scores.append(score_data["order_accuracy"]["normalized_kendall_tau"])
            average_semantic_scores.append(score_data["sem_score_match_stats"]["mean"])
            count_accuracy_scores.append(score_data["count_accuracy"])
            overall_performance_scores.append(score_data["final_score"])

        # Calculate statistics
        def compute_statistics(data):
            return {
                "mean": np.mean(data),
                "std": np.std(data),
                "min": np.min(data),
                "max": np.max(data),
                "count": len(data)
            }

        # Calculate overall statistics
        order_accuracy_statistics = compute_statistics(order_accuracy_scores)
        semantic_score_statistics = compute_statistics(average_semantic_scores)
        count_accuracy_statistics = compute_statistics(count_accuracy_scores)
        overall_performance_statistics = compute_statistics(overall_performance_scores)

        # Store the overall statistics
        aggregate_stats = {
            "order_accuracy_score_stats": order_accuracy_statistics,
            "matched_name_semantic_score_stats": semantic_score_statistics,
            "count_accuracy_score_stats": count_accuracy_statistics,
            "final_score_stats": overall_performance_statistics
        }

        return aggregate_stats
    else:
        aggregate_stats = {}
        
def default_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def generate_json_output(scores, output_file):
    # Format the start time for the analysis
    analysis_start_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Compute aggregate statistics from scores
    aggregate_stats = compute_aggregate_stats(scores)
    
    # Compile all the relevant data into a dictionary
    output_data = {
        "analysis_start_time": analysis_start_time,
        "scores": scores,
        "aggregate_stats": aggregate_stats
    }
    
    # Write the compiled data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4, default=default_converter)
    
    # Change file ownership to host user if script is running as root in docker
    if os.getuid() == 0:
            host_uid = 1000  # Replace with actual host user ID
            host_gid = 1000  # Replace with actual host group ID
            os.chown(output_file, host_uid, host_gid)
            
    print(f"Output written to {output_file}")

def main(model_responses_file, gt_values_file):
    model_responses = load_json_file(model_responses_file)
    gt_values = load_json_file(gt_values_file)
    embed_model = TextEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    output_file = f"response_scores_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    scores = {}  # This will store the evaluation results similarly to 'results'
    for filename, response_data in model_responses.items():
        response_names = parse_responses(response_data['response'])
        gt_names = gt_values.get(filename, {}).get('objects_left_to_right', [])
        evaluation_result = evaluate_outputs(response_names, gt_names, embed_model)
        scores[filename] = evaluation_result

    # Generate output JSON file with results and statistics
    generate_json_output(scores, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model responses against ground truth.")
    parser.add_argument('model_responses_file', type=str, help='Path to the model responses JSON file.')
    parser.add_argument('gt_values_file', type=str, help='Path to the ground truth JSON file.')
    args = parser.parse_args()

    main(args.model_responses_file, args.gt_values_file)
