import datetime as dt
import os
import json
import sys
import base64
import argparse
import logging
import ollama
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fastembed import TextEmbedding
from dotenv import load_dotenv
from html import escape
from scipy.stats import kendalltau

# Configure logging and environment
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("roboai")
load_dotenv("shared/.env")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process image dir and compare LLM captions.")
    parser.add_argument("dir_name", type=str, nargs="?", help="Name of the dir containing the dataset. If not specified, the script will use the 'latest' symlink.")
    parser.add_argument("-m", "--model", type=str, default="llava:latest", nargs="?", help="Name of ollama model. Include tag; eg. 'moondream:v2'. (default: 'llava:latest')")
    return parser.parse_args()

def get_model_info(model_name):
    """combines data from ollama list and show calls to create detailed model info"""
    model_list_data = ollama.list()
    model_data_entry = next((model for model in model_list_data['models'] if model['name'] == model_name), None)
    
    if not model_data_entry:
        logger.error(f"Model {model_name} not found in the list.")
        return None

    additional_data = ollama.show(model_name)
    
    model_info = {
        "name": model_data_entry.get("name"),
        "model": model_data_entry.get("model"),
        "modified_at": model_data_entry.get("modified_at"),
        "size": model_data_entry.get("size"),
        "digest": model_data_entry.get("digest"),
        "details": model_data_entry.get("details", {}),
        "modelfile": additional_data.get("modelfile"),
        "parameters": additional_data.get("parameters"),
        "template": additional_data.get("template")
    }
    
    return model_info

def preload_model(model_name):
    try:
        preload_response = ollama.generate(model=model_name, keep_alive=-1)
        print("Model preloaded successfully.\n", preload_response)
    except ollama.ResponseError as e:
        logger.error(f"ollama Response Error during preload: {e.error}")
    except Exception as err:
        logger.error(f"Unexpected error during model preload: {err}")

def unload_model(model_name):
    try:
        unload_response = ollama.generate(model=model_name, keep_alive=0)
        print("Model unloaded successfully.\n", unload_response)
    except ollama.ResponseError as e:
        logger.error(f"ollama Response Error during unload: {e.error}")
    except Exception as err:
        logger.error(f"Unexpected error during model unload: {err}")

def encode_image_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

def get_caption_with_image(image_path, model_name, prompt):
    try:
        image_base64 = encode_image_to_base64(image_path)
        # response = ollama.generate(model=model_name, prompt=prompt, stream=False, images=[image_base64], format="json")
        response = ollama.generate(model=model_name, prompt=prompt, stream=False, images=[image_base64])
        return response
    except ollama.ResponseError as e:
        logger.error(f"ollama Response Error: {e.error}")
        return {"error": "ollama Response Error", "status_code": e.status_code, "details": e.error}
    except Exception as err:
        logger.error(f"An unexpected error occurred: {err}")  # Log unexpected errors
        return {"error": "Unexpected error", "details": str(err)}

def cosine_similarity_np(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return float(dot_product / (norm_vec1 * norm_vec2))

def normalize_text(data):
    """Normalize data to be a JSON string if it's a dictionary."""
    if isinstance(data, dict):
        return json.dumps(data, separators=(', ', ': '))  # Convert dictionary to JSON string
    if isinstance(data, str):
        try:
            json_obj = json.loads(data)  # Attempt to load string as JSON
            return json.dumps(json_obj, separators=(', ', ': '))  # Convert back to normalized string
        except json.JSONDecodeError:
            pass  # Not a JSON string, treat it as a plain string
    return data

def get_dir_path(latest_symlink_path, dir_name=None):
    print(f"Latest symlink path: {latest_symlink_path}")
    print(f"Directory name: {dir_name}")
    
    if dir_name:
        dir_path = os.path.join("/app/shared/data/image_exports", dir_name)
    else:
        try:
            resolved_symlink = os.path.join(os.path.dirname(latest_symlink_path), os.readlink(latest_symlink_path))
            print(f"Resolved symlink '{latest_symlink_path}' to directory '{resolved_symlink}'")
            dir_path = resolved_symlink
        except OSError as e:
            print(f"Failed to read symlink {latest_symlink_path}: {e}")
            sys.exit(1)
            
    print(f"Final directory path: {dir_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory exists: {os.path.isdir(dir_path)}")
    
    if not os.path.isdir(dir_path):
        print(f"Directory '{dir_path}' not found.")
        sys.exit(1)
    
    return dir_path

def process_dir(dir_path, model_name, prompt):
    sim_metadata_path = os.path.join(dir_path, f"{os.path.basename(dir_path)}.json")
    with open(sim_metadata_path, 'r') as file:
        sim_metadata = json.load(file)

    preload_model(model_name)

    responses = {}
    validation_data = {}
    for filename in sorted(os.listdir(dir_path)):
        if filename.endswith(".png"):
            image_path = os.path.join(dir_path, filename)
            flip = "_flipped" in filename
            base_filename = filename.replace('_flipped', '')

            metadata_entry = next((item for item in sim_metadata['runs'] if item['image_file'] == base_filename), None)
            if not metadata_entry:
                print(f"No metadata found for {filename}")
                continue

            api_response = get_caption_with_image(image_path, model_name, prompt)
            responses[filename] = api_response

            if 'response' in api_response:
                print(f"Processed {filename}: {api_response.get('response', 'No content')}")
            elif 'message' in api_response:
                message_content = api_response['message'].get('content', 'No content') if isinstance(api_response['message'], dict) else 'No content'
                print(f"Processed {filename}: {message_content}")
            else:
                error_message = f"Error processing {filename}: {api_response.get('error', 'Unknown error')}"
                print(error_message)
                
            augmentation_map = {
                "Milk": "White milk carton",
                "Can": "Red soda can",
                "Cereal": "Red cereal box",
                "Bread": "Brown bread loaf"
            }

            if metadata_entry:
                sorted_names = [augmentation_map.get(name, name) for name in sort_objects_by_leftward_plane(metadata_entry['objects'], flip=flip)]
                validation_data[filename] = {
                    "objects": {
                        "count": len(metadata_entry['objects']),
                        "names": sorted_names
                    }
                }

    unload_model(model_name)

    scores = {}
    if model_name != "moondream:v2":
        embed_model = TextEmbedding("mixedbread-ai/mxbai-embed-large-v1")
        for filename, response in responses.items():
            if 'response' in response:
                response_obj = json.loads(response['response'])
                validation_obj = validation_data.get(filename, {})

                response_names = response_obj['objects']['names']
                response_count = response_obj['objects']['count']
                gt_names = validation_obj['objects']['names']
                gt_count = validation_obj['objects']['count']

                scores[filename] = evaluate_outputs(response_names, gt_names, embed_model, response_count, gt_count)
                
    return responses, sim_metadata, validation_data, scores

def sort_objects_by_leftward_plane(objects, flip=False):
    if flip:  # flipped images have to use rightward values rather than reverse order
        sorted_objects = sorted(objects, key=lambda x: x['distances']['rightward_plane'])
    else:
        sorted_objects = sorted(objects, key=lambda x: x['distances']['leftward_plane'])
    sorted_names = [obj['name'] for obj in sorted_objects]
    return sorted_names

def calculate_semantic_similarity(response_names, gt_names, embed_model, similarity_threshold=0.0):
    """
    For each response name, find the ground truth name with the highest cosine similarity.
    Any response names that do not have a matching ground truth are reported as unpaired,
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
        "mean": np.mean(all_cosine_similarities),
        "std": np.std(all_cosine_similarities),
        "min": np.min(all_cosine_similarities),
        "max": np.max(all_cosine_similarities),
        "count": len(all_cosine_similarities)
    }

    # Calculate statistics for sem_score_matches
    sem_score_match_stats = {
        "mean": np.mean([pair['cosine_similarity'] for pair in sem_score_matches]),
        "std": np.std([pair['cosine_similarity'] for pair in sem_score_matches]),
        "min": min([pair['cosine_similarity'] for pair in sem_score_matches], default=0),
        "max": max([pair['cosine_similarity'] for pair in sem_score_matches], default=0),
        "count": len(sem_score_matches)
    }
    
    return sem_score_matches, unpaired_responses, sem_score_match_stats, all_sem_scores, all_sem_score_stats

def calculate_count_accuracy(response_count, gt_count, response_len):
    # Internal consistency is based on the relative difference between the count and the length of the names list
    if response_len > 0:
        internal_consistency = 1 - (abs(response_count - response_len) / response_len)
    else:
        internal_consistency = 0 if response_count != 0 else 1  # Both are zero

    # Calculate the count score based on the difference between the response and ground truth counts
    if gt_count != 0:
        count_score = 1 - (abs(response_count - gt_count) / gt_count)
    else:
        count_score = 0 if response_count != 0 else 1  # Both are zero

    return max(0, internal_consistency * count_score)

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
        "normalized_kendall_tau": normalized_tau,
        "p_value": p_value,
        "matched_indices": matched_indices,
        "response_indices": response_indices
    }

def evaluate_outputs(response_names, gt_names, embed_model, response_count, gt_count):
    sem_score_matches, unpaired_responses, sem_score_match_stats, all_sem_scores, all_sem_score_stats = calculate_semantic_similarity(response_names, gt_names, embed_model)
    order_accuracy_result = kendall_tau_normalized(sem_score_matches)
    count_accuracy_score = calculate_count_accuracy(response_count, gt_count, len(response_names))

    # Weight factors
    weight_order = 0.1
    weight_count = 0.2
    weight_similarity = 0.7

    # Final score calculation
    final_score = (weight_order * order_accuracy_result["normalized_kendall_tau"] +
                   weight_count * count_accuracy_score +
                   weight_similarity * sem_score_match_stats["mean"])

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

def compare_results(responses):
    # Placeholder for comparison logic
    print("Comparison results (placeholder):", responses)

def generate_json_output(responses, sim_metadata, model_info, prompt, validation_data, scores, output_file='output.json'):
    analysis_start_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    aggregate_stats = compute_aggregate_stats(scores)
        
    output_data = {
        "sim_info": sim_metadata,
        "model_info": model_info,
        "analysis_start_time": analysis_start_time,
        "prompt": prompt,
        "responses": responses,
        "validation_data": validation_data,
        "scores": scores,
        "aggregate_stats": aggregate_stats
    }

    # Debugging: Log each item's type in results to check for non-serializable objects
    for key, value in responses.items():
        logger.debug(f"Key: {key}, Type of value: {type(value)}")

    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)
    logger.info(f"JSON output generated at {output_file}")

def extract_data(data):
    return {
        "responses": data['responses'],
        "sim_info": data['sim_info']['runs'],
        "validation_data": data['validation_data'],
        "scores": data['scores'],
        "model_info": data['model_info'],
        "aggregate_stats": json.dumps(data.get('aggregate_stats', {}), indent=4).replace('\n', '<br>')
    }

def generate_histogram(data, title, xlabel, ylabel, output_file):
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    n = len(data)
    plt.text(0.95, 0.95, f'n = {n}', ha='right', va='top', transform=plt.gca().transAxes)
    plt.savefig(output_file)
    plt.close()
    
def generate_graphs(scores, graph_dir):
    sem_scores_output_file = os.path.join(graph_dir, 'sem_scores_histogram.png')
    final_scores_output_file = os.path.join(graph_dir, 'final_scores_histogram.png')
    
    # Extract data for histograms
    avg_sem_scores_for_histogram = [item['sem_score_match_stats']['mean'] for item in scores.values() if 'sem_score_match_stats' in item]
    final_scores_for_histogram = [item['final_score'] for item in scores.values() if 'final_score' in item] 

    # Generate histograms
    if avg_sem_scores_for_histogram:
        generate_histogram(
            avg_sem_scores_for_histogram,
            title='Matched Name Avg SemScores',
            xlabel='SemScore',
            ylabel='Frequency',
            output_file=sem_scores_output_file
        )
    if final_scores_for_histogram:
        generate_histogram(
            final_scores_for_histogram,
            title='Final Scores',
            xlabel='Score',
            ylabel='Frequency',
            output_file=final_scores_output_file
        )

def html_from_output_json(json_file_path, html_output_path):
    try:
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        
        extracted_data = extract_data(data)
        responses = extracted_data['responses']
        validation_data = extracted_data['validation_data']
        scores = extracted_data['scores']
        sim_info = extracted_data['sim_info']
        model_info = extracted_data['model_info']
        
        rows = []
        for filename, content in responses.items():
            formatted_prompt = escape(data['prompt']).replace("\n", "<br>")
            prompt_details = f'<details><summary>View Prompt</summary><code class="prompt">{formatted_prompt}</code></details>'

            response = escape(content.get('response', '')).replace("\n", "<br>")

            per_file_validation_data = validation_data.get(filename, {})
            per_file_validation_details = json.dumps(per_file_validation_data, indent=4).replace("\n", "<br>")
            
            score_data = scores.get(filename, {})            
            per_file_matched_names = [
                json.dumps(score_data.get('sem_score_matches', []), indent=4).replace("\n", "<br>"),
                json.dumps(score_data.get('unpaired_responses', []), indent=4).replace("\n", "<br>")
            ]
            per_file_matched_names_formatted = f"<details><summary>View Details</summary><pre><code>{''.join(per_file_matched_names)}</code></pre></details>"
            per_file_all_sem_scores = json.dumps(score_data.get('all_sem_scores', {}), indent=4).replace("\n", "<br>")
            per_file_all_sem_scores_formatted = f"<details><summary>View Details</summary><pre><code>{per_file_all_sem_scores}</code></pre></details>"
            per_file_all_sem_score_stats = json.dumps(score_data.get('all_sem_score_stats', {}), indent=4).replace("\n", "<br>")
            per_file_all_sem_score_stats_formatted = f"<details><summary>View Details</summary><pre><code>{per_file_all_sem_score_stats}</code></pre></details>"

            full_response_json = json.dumps(content, indent=4).replace("\n", "<br>")
            full_response_formatted = f"<details><summary>View Details</summary><pre><code>{full_response_json}</code></pre></details>"

            sim_details = next((run for run in sim_info if run['image_file'] == filename), None)
            sim_details_json = json.dumps(sim_details, indent=4).replace("\n", "<br>") if sim_details else "No sim details available"
            sim_details_formatted = f"<details><summary>View Sim Details</summary><pre><code>{sim_details_json}</code></pre></details>"
            
            model_info_json = json.dumps(model_info, indent=4).replace("\n", "<br>")
            model_info_formatted = f"<details><summary>View Model Info</summary><pre><code>{model_info_json}</code></pre></details>"
            model_name_table = model_info.get('name', {})

            rows.append({
                'Run': filename.split('.')[0],
                'Image Filename': filename,
                'Image': f'<img src="../../{filename}" alt="{filename}" class="expandable">',
                'Model Name': model_name_table,
                'Prompt': prompt_details,
                'Response Message': f'<code class="response">{response}</code>',
                'Validation Data': f'<code class="gt">{per_file_validation_details}</code>',
                'Final Score': score_data.get('final_score', ''),
                'Matched Name Avg SemScore': score_data.get('sem_score_match_stats', {}).get('mean', ''),
                'Count Accuracy': score_data.get('count_accuracy', ''),
                'Order Accuracy': score_data.get('order_accuracy', {}).get('normalized_kendall_tau', ''),
                'Matches & Unpaired Responses': per_file_matched_names_formatted,
                'All SemScores': per_file_all_sem_scores_formatted,
                'All SemScore Stats': per_file_all_sem_score_stats_formatted, 
                'Full Response': full_response_formatted,
                'Sim Details': sim_details_formatted,
                'Model Info': model_info_formatted
            })
        df = pd.DataFrame(rows)

        html_style_script = '''
        <style>
            body { background-color: #263238; color: #ECEFF1; font-family: monospace; font-size: 1.1em; }
            pre { white-space: pre-wrap; font-size: smaller;}
            code { white-space: pre-wrap; background-color: black; padding: 4px; display: block; min-width: 80ch}
            code.prompt { color: #80d4ff }
            code.response { color: #66ff66 }
            code.gt { color: #ffa366 }
            details summary { cursor: pointer; }
            img.expandable {width: 100px; height: auto; cursor: pointer; transition: transform 0.25s ease; }
            img.expandable:hover { transform: scale(1.05); }
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid #37474F; padding: 12px; text-align: left; font-size: 0.9em; overflow: visible; position: relative; }
            th.sortable:hover { cursor: pointer; }
        </style>
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script>
            $(document).ready(function() {
                var table = $('table');
                var sortedAscending = true;

                // Function to set sorting indicator
                function setSortingIndicator(header, ascending) {
                    table.find('th').each(function() {
                        $(this).find('.sort-indicator').remove();
                    });

                    var indicator = ascending ? ' ▲' : ' ▼';
                    header.append('<span class="sort-indicator">' + indicator + '</span>');
                }

                function compareFilenames(a, b) {
                    const extractNumber = filename => parseInt(filename.match(/(\d+)/), 10);
                    const aNumber = extractNumber(a);
                    const bNumber = extractNumber(b);

                    if (aNumber !== bNumber) {
                        return aNumber - bNumber;
                    }

                    const aFlipped = a.includes("_flipped");
                    const bFlipped = b.includes("_flipped");

                    return aFlipped - bFlipped; // True (1) will follow False (0)
                }

                function sortTable(columnIndex, ascending, comparator) {
                    var rows = table.find('tbody tr').toArray();
                    rows.sort(function(a, b) {
                        var aVal = $(a).children('td').eq(columnIndex).text().trim();
                        var bVal = $(b).children('td').eq(columnIndex).text().trim();

                        if (comparator) {
                            return ascending ? comparator(aVal, bVal) : comparator(bVal, aVal);
                        }

                        var aValNum = parseFloat(aVal) || 0;
                        var bValNum = parseFloat(bVal) || 0;
                        return ascending ? aValNum - bValNum : bValNum - aValNum;
                    });

                    $.each(rows, function(index, row) {
                        table.children('tbody').append(row);
                    });
                }

                // Adding click event to the semscore and Run headers
                table.find('th.sortable').on('click', function() {
                    var columnIndex = $(this).index();
                    sortedAscending = !sortedAscending;
                    var comparator = columnIndex === 0 ? compareFilenames : null; // Custom comparator for filenames
                    sortTable(columnIndex, sortedAscending, comparator);

                    // Update the sorting indicator
                    setSortingIndicator($(this), sortedAscending);
                });

                // Handle expandable images
                document.querySelectorAll('img.expandable').forEach(img => {
                    img.onclick = function () {
                        const cell = img.closest('td');
                        if (img.style.width === '100px' || img.style.width === '') {
                            img.style.position = 'absolute';
                            img.style.width = 'auto';
                            img.style.maxWidth = '100%';
                            img.style.zIndex = '1000';
                            cell.style.position = 'static';
                        } else {
                            img.style.position = '';
                            img.style.width = '100px';
                            img.style.maxWidth = '';
                            img.style.zIndex = '';
                        }
                    };
                });
            });
        </script>
        '''

        html_table = df.to_html(escape=False, index=False)
        for header in ['Run', 'Final Score', 'Matched Name Avg SemScore', 'Count Accuracy', 'Order Accuracy']:
            html_table = html_table.replace(f'<th>{header}</th>', f'<th class="sortable">{header}</th>')
        
        aggregate_stats_formatted = f"<details><summary>View Aggregate Stats</summary><pre><code>{extracted_data['aggregate_stats']}</code></pre></details>"
        final_scores_histogram = f'<details><summary>View Final Score Histogram</summary><img src="graphs/final_scores_histogram.png" alt="Final Scores Histogram"></details>'
        sem_scores_histogram = f'<details><summary>View Matched Name Avg SemScore Histogram</summary><img src="graphs/sem_scores_histogram.png" alt="SemScore Histogram"></details>'
        html_content = f"<head>{html_style_script}</head><body>{aggregate_stats_formatted}{final_scores_histogram}{sem_scores_histogram}{html_table}</body>"

        with open(html_output_path, 'w') as f:
            f.write(html_content)

        print(f"HTML output generated at {html_output_path}")

    except Exception as e:
        print(f"An error occurred while generating HTML: {e}")

def main():
    args = parse_arguments()

    model_name = args.model
    prompt = "Name all the objects on the table in order from left to right as a list of strings."
#     prompt = """How many objects are on the table? What are the objects? Order the objects as they appear from left to right.
    
# Respond using JSON. Follow the pattern: 
# {
#     "objects": {
#         "count": <number_of_objects>,
#         "names": [
#             "first object from left",
#             "next object from left", 
#             "last object from left"
#         ]
#     }
# }"""

    latest_symlink_path = "/app/shared/data/image_exports/latest"
    dir_path = get_dir_path(latest_symlink_path, args.dir_name)

    analysis_dir = os.path.join(dir_path, "analysis")
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)
        os.chmod(analysis_dir, 0o755)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name_dir = model_name.replace(':', '-')
    analysis_session_dir = os.path.join(analysis_dir, f"{timestamp}_{model_name_dir}")
    graphs_dir = os.path.join(analysis_session_dir, 'graphs')
    os.makedirs(analysis_session_dir)
    os.makedirs(graphs_dir)
    os.chmod(analysis_session_dir, 0o755)
    os.chmod(graphs_dir, 0o755)

    model_info = get_model_info(model_name)
    if model_info:
        logger.info(f"Model Info: {model_info}")
        responses, sim_metadata, validation_data, scores = process_dir(dir_path, model_name, prompt)
        compare_results(responses)

        output_json_path = os.path.join(analysis_session_dir, 'output.json')
        output_html_path = os.path.join(analysis_session_dir, 'output.html')
        generate_json_output(responses, sim_metadata, model_info, prompt, validation_data, scores, output_file=output_json_path)
        html_from_output_json(output_json_path, output_html_path)
        generate_graphs(scores, graphs_dir)

        # Set file permissions for files
        os.chmod(output_json_path, 0o644)
        os.chmod(output_html_path, 0o644)
        for file in os.listdir(graphs_dir):
            graph_path = os.path.join(graphs_dir, file)
            if os.path.exists(graph_path):
                os.chmod(graph_path, 0o644)

        # Optionally change ownership to match the host user, if the script runs as root
        if os.getuid() == 0:
            host_uid = 1000  # Replace with actual host user ID
            host_gid = 1000  # Replace with actual host group ID
            os.chown(analysis_dir, host_uid, host_gid)
            os.chown(analysis_session_dir, host_uid, host_gid)
            os.chown(graphs_dir, host_uid, host_gid)
            os.chown(output_json_path, host_uid, host_gid)
            os.chown(output_html_path, host_uid, host_gid)
            for file in os.listdir(graphs_dir):
                graph_path = os.path.join(graphs_dir, file)
                if os.path.exists(graph_path):
                    os.chown(graph_path, host_uid, host_gid)
    else:
        logger.error("Failed to fetch model information. Aborting image processing.")


if __name__ == "__main__":
    main()
