import argparse
import datetime as dt
import json
import os
import numpy as np
from scipy.stats import kendalltau
from fastembed import TextEmbedding

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model responses against ground truth.")
    parser.add_argument('model_responses_file', type=str, help='Path to the model responses JSON file.')
    parser.add_argument('gt_values_file', type=str, help='Path to the ground truth JSON file.')
    return parser.parse_args()

def load_json_file(filepath):
    """
    Load JSON data from a file.
    """
    with open(filepath, 'r') as file:
        return json.load(file)

def set_clean_responses_model(model_id):
    """
    Returns the appropriate cleaning function based on the model_id.
    """
    if model_id == 'vikhyatk/moondream2':
        return clean_responses_moondream2
    elif model_id == 'HuggingFaceM4/idefics2-8b-chatty':
        return clean_responses_idefics2
    elif model_id == 'paligemma-3b-mix-448':
        return clean_responses_paligemma
    elif model_id == 'llava:latest':
        return clean_responses_llava
    else:
        raise RuntimeError(f"Input json is incompatible or uses an unsupported model. Provided model: {model_id}")

def clean_responses_moondream2(response_string):
    """
    Parse moondream2 response strings and isolate object names by splitting on commas,
    also removing periods and the word 'and' following oxford comma.
    """
    items = response_string.replace('.', '').replace(', and', ',').split(', ')
    return [item.strip() for item in items if item]

def clean_responses_idefics2(response_string):
    """
    Parse idecfics2 response strings and isolate object names by removing prefixed text up to 'are',
    then splitting on commas, and also removing periods and the word 'and' following oxford comma.
    """
    # Find the position of ' are' and slice the string from that position plus the length of ' are'
    start_idx = response_string.find(' are') + len(' are')
    processed_string = response_string[start_idx:]
    
    # Clean and split the string as before
    items = processed_string.replace('.', '').replace(', and', ',').split(', ')
    return [item.strip() for item in items if item]

def clean_responses_llava(response_string):
    """
    Parse llava response strings and isolate object names from json format.
    """
    items = response_string.replace('.', '').replace(', and', ',').split(', ')
    return [item.strip() for item in items if item]

def clean_responses_paligemma(response_string):
    """
    Parse paligemma response strings and isolate object names by splitting on commas,
    also removing periods and the word 'and' following oxford comma
    """
    items = response_string.replace('.', '').replace(', and', ',').split(', ')
    return [item.strip() for item in items if item]

def cosine_similarity_np(vec1, vec2):
    """
    Compute cosine similarity between two vectors using numpy.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_semantic_similarity(response_names, gt_names, embed_model, similarity_threshold=0.50):
    """
    For each response name, find the ground truth name with the highest cosine similarity between
    embedded strings (semscore).
    
    Any response names that do not have a matching ground truth >0.50 are reported as unpaired,
    usually when len(response_names) > len(gt_names). 
    
    Returns best matches (with stats), unpaired responses, and all semscore comparisons (with stats)
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
    if not all_cosine_similarities:  # Check if the array is empty
        all_sem_score_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    else:
        all_sem_score_stats = {
            "mean": float(np.mean(all_cosine_similarities)),
            "std": float(np.std(all_cosine_similarities)),
            "min": float(np.min(all_cosine_similarities)),
            "max": float(np.max(all_cosine_similarities)),
            "count": len(all_cosine_similarities)
        }

    if not sem_score_matches:  # Check if no matches were found
        sem_score_match_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0}
    else:
        sem_score_match_stats = {
            "mean": float(np.mean([pair['cosine_similarity'] for pair in sem_score_matches])),
            "std": float(np.std([pair['cosine_similarity'] for pair in sem_score_matches])),
            "min": float(min([pair['cosine_similarity'] for pair in sem_score_matches], default=0)),
            "max": float(max([pair['cosine_similarity'] for pair in sem_score_matches], default=0)),
            "count": len(sem_score_matches)
        }
    return sem_score_matches, unpaired_responses, sem_score_match_stats, all_sem_scores, all_sem_score_stats

def calculate_count_accuracy(response_count, gt_count):
    """
    Calculate the count score based on the difference between the response and ground truth counts
    """
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
    
    # Handling single element or no element scenarios
    if len(matched_indices) < 2 or len(response_indices) < 2:
        if len(matched_indices) == 1 and len(response_indices) == 1:
            # Single match, could consider it as perfect or undefined.
            return {
                "normalized_kendall_tau": 1.0,  # Consider perfect correlation if only one matched pair exists
                "p_value": 0.0,
                "matched_indices": matched_indices,
                "response_indices": response_indices
            }
        else:
            # Not enough data to calculate Kendall's tau, return NaN
            return {
                "normalized_kendall_tau": 0.0,
                "p_value": 1.0,
                "matched_indices": matched_indices,
                "response_indices": response_indices,
                "reason": "Insufficient data for Kendall's tau calculation"
            }
    
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
    """
    Scores responses vs gt values on object name semscore (embedded string cosine similarity),
    object ordering accuracy, object count accuracy.
    """
    sem_score_matches, unpaired_responses, sem_score_match_stats, all_sem_scores, all_sem_score_stats = calculate_semantic_similarity(response_names, gt_names, embed_model)
    order_accuracy_result = kendall_tau_normalized(sem_score_matches)
    count_accuracy_score = calculate_count_accuracy(len(response_names), len(gt_names))
    
    mean_sem_score = sem_score_match_stats["mean"] if sem_score_match_stats["mean"] is not None else 0
    normalized_kendall_tau = order_accuracy_result["normalized_kendall_tau"] if order_accuracy_result["normalized_kendall_tau"] is not None else 0
    
    final_score = (normalized_kendall_tau * count_accuracy_score * mean_sem_score)
    
    return {
        "response_names": response_names,
        "gt_names": gt_names,
        "count_accuracy": count_accuracy_score,
        "order_accuracy": order_accuracy_result,
        "sem_score_matches": sem_score_matches,
        "unpaired_responses": unpaired_responses,
        "sem_score_match_stats": sem_score_match_stats,
        "all_sem_scores": all_sem_scores,
        "all_sem_score_stats": all_sem_score_stats,
        "final_score": final_score
    }

def compute_statistics(data):
    # Filter out None values from the data list
    filtered_data = [x for x in data if x is not None]
    
    if not filtered_data:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}  # Return 0 or another appropriate value if all data is None or list is empty

    return {
        "mean": np.mean(filtered_data),
        "std": np.std(filtered_data),
        "min": np.min(filtered_data),
        "max": np.max(filtered_data),
        "count": len(filtered_data)
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
    """
    For cleaning json data types when needed
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert numpy arrays to lists
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def generate_json_output(model_id, model_responses_file, gt_values_file, scores, output_file):
    """
    Outputs a json file with scores for model reponses vs gt
    """
    # Format the start time for the analysis
    analysis_start_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Compute aggregate statistics from scores
    aggregate_stats = compute_aggregate_stats(scores)
    
    # Compile all the relevant data into a dictionary
    output_data = {
        "analysis_start_time": analysis_start_time,
        "model_id" : model_id,
        "model_json": model_responses_file,
        "gt_json": gt_values_file,
        "scores": scores,
        "aggregate_stats": aggregate_stats
    }
    
    # Write the compiled data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=4, default=default_converter)
    
    # Change file ownership to host user if script is running as root in docker
    adjust_file_permissions(output_file)
            
    print(f"Output written to {output_file}")
    
def adjust_file_permissions(output_file):
    """
    Fixes file permission when running script as root in docker
    """
    if os.getuid() == 0:  # Running as root
        host_uid, host_gid = 1000, 1000  # Default IDs, adjust as necessary
        os.chown(output_file, host_uid, host_gid)
        print(f"File ownership changed for {output_file}")

def compare_responses_to_gt(model_responses, gt_json, embed_model, clean_responses_func):
    """
    Parses and extracts strings for object names in responses and gt values, then sends them
    through the scoring evalution process
    """
    scores = {}
    for filename, response_data in model_responses.items():
        response_names = clean_responses_func(response_data['response'])
        gt_names = gt_json.get(filename, {}).get('objects_left_to_right', [])
        evaluation_result = evaluate_outputs(response_names, gt_names, embed_model)
        scores[filename] = evaluation_result
    return scores

def main(model_responses_file, gt_values_file):
    model_json = load_json_file(model_responses_file)
    model_id = model_json['model_id']
    clean_response_func = set_clean_responses_model(model_id)
    model_id_cleaned = model_id.replace(':', '-').replace('/', '_')
    model_responses = model_json['responses']
    gt_json = load_json_file(gt_values_file)
    embed_model = TextEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    output_file = f"response_scores_{model_id_cleaned}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    scores = compare_responses_to_gt(model_responses, gt_json, embed_model, clean_response_func)

    # Generate output JSON file with results and statistics
    generate_json_output(model_id, model_responses_file, gt_values_file, scores, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.model_responses_file, args.gt_values_file)
