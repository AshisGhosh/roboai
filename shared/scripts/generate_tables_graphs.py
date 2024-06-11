# python ../../../scripts/generate_tables_graphs_nostylescript.py response_scores_HuggingFaceM4_idefics2-8b-chatty_20240608_031525.json

import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io


def parse_arguments():
    parser = argparse.ArgumentParser(description="Convert model score jsons to html table view and create graphs.")
    parser.add_argument("scores_json_files", type=str, nargs="+", help="response score json files")
    return parser.parse_args()

def load_data(file_paths):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            data.append(json.load(file))
    return data

def plot_histograms_svg(data, title, xlabel, ylabel, xlim=None, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = 50  # Increase the number of bins for thinner bars
    ax.hist(data, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=14, labelpad=15)
    ax.set_ylabel(ylabel, fontsize=14, labelpad=15)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', length=4, color='gray')
    ax.minorticks_on()
    ax.grid(True, linestyle='--', alpha=0.5, which='both')

    n = len(data)
    max_height = max(np.histogram(data, bins=bins)[0]) * 1.1  # Add 10% headroom
    if ylim:
        ylim = (ylim[0], max(max_height, ylim[1]))
    else:
        ylim = (0, max_height)

    ax.text(0.95, 0.95, f'n = {n}', ha='right', va='top', transform=ax.transAxes, fontsize=12)
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    
    svg_file = io.StringIO()
    plt.savefig(svg_file, format='svg', bbox_inches='tight')
    plt.close(fig)
    svg_file.seek(0)
    return svg_file.getvalue()

def compute_limits(data, metrics):
    limits = {}
    for metric_name, aggregate_key, score_key, chart_title in metrics:
        values = []
        for model_data in data:
            if score_key == 'sem_score_match_stats':
                values.extend([info[score_key]['mean'] for filename, info in model_data['scores'].items()])
            elif score_key == 'order_accuracy':
                values.extend([info[score_key]['normalized_kendall_tau'] for filename, info in model_data['scores'].items()])
            else:
                values.extend([info[score_key] for filename, info in model_data['scores'].items()])
        
        if values:
            limits[score_key] = (min(values), max(values))
    return limits

def compute_max_frequencies(data, metrics):
    max_frequencies = {}
    for metric_name, aggregate_key, score_key, chart_title in metrics:
        max_freq = 0
        for model_data in data:
            if score_key == 'sem_score_match_stats':
                values = [info[score_key]['mean'] for filename, info in model_data['scores'].items()]
            elif score_key == 'order_accuracy':
                values = [info[score_key]['normalized_kendall_tau'] for filename, info in model_data['scores'].items()]
            else:
                values = [info[score_key] for filename, info in model_data['scores'].items()]
            
            freq, _ = np.histogram(values, bins=20)
            max_freq = max(max_freq, max(freq))
        max_frequencies[score_key] = max_freq
    return max_frequencies

def generate_aggregate_stats_table(stats):
    html_content = '<table border="1">'
    html_content += f'<tr><td>Mean</td><td>{stats["mean"]}</td></tr>'
    html_content += f'<tr><td>Std</td><td>{stats["std"]}</td></tr>'
    html_content += f'<tr><td>Min</td><td>{stats["min"]}</td></tr>'
    html_content += f'<tr><td>Max</td><td>{stats["max"]}</td></tr>'
    html_content += f'<tr><td>Count</td><td>{stats["count"]}</td></tr>'
    html_content += '</table>'
    return html_content

def generate_detailed_model_scores(model_data):
    details_list = []
    for filename, info in model_data['scores'].items():
        matches_unpaired = json.dumps({"Matches": info["sem_score_matches"], "Unpaired": info["unpaired_responses"]}, indent=4).replace("\n", "<br>")
        all_sem_scores = json.dumps(info["all_sem_scores"], indent=4).replace("\n", "<br>")

        details = {
            'Image Filename': f'{filename}',
            'Prompt': f'<code class="prompt">{model_data["prompt"]}</code>',
            'Response objects': f'<code class="response">{", ".join(info["response_names"])}</code>',
            'GT objects': f'<code class="gt">{", ".join(info["gt_names"])}</code>',
            'Final Score': f'{info["final_score"]:.3f}',
            'Matched Name Avg SemScore': f'{info["sem_score_match_stats"]["mean"]:.3f}',
            'Count Accuracy': f'{info["count_accuracy"]:.3f}',
            'Order Accuracy': f'{info["order_accuracy"]["normalized_kendall_tau"]:.3f}',
            'Matches & Unpaired Responses': f'<details><summary>Click to view</summary><pre><code>{matches_unpaired}</pre></details>',
            'All SemScores': f'<details><summary>Click to view</summary><pre><code>{matches_unpaired}</code></pre></details>'
        }
        details_list.append(details)
    
    details_df = pd.DataFrame(details_list)
    model_id = model_data['model_id']
    html_content = f'<div id="details-{model_id}" class="model-details"><h3>Detailed Model Scores for Model {model_id}</h3>'
    
    html_content += details_df.to_html(index=False, escape=False)
    html_content += '</div>'
    return html_content

def generate_histogram(score_data, chart_title, limits, max_frequencies, score_key):
    xlim, ylim = limits[score_key], (0, max_frequencies[score_key])
    return plot_histograms_svg(score_data, chart_title, 'Score', 'Frequency', xlim, ylim)

def generate_model_overview(model_data, metrics, limits, max_frequencies):
    html_content = f'<div class="model-aggregate-stats"><h3>Aggregate Statistics for Model {model_data["model_id"]}</h3>'
    aggregate_stats = model_data['aggregate_stats']
    
    for metric_name, aggregate_key, score_key, chart_title in metrics:
        stats = aggregate_stats[aggregate_key]
        html_content += f'<div class="metric"><h4>{metric_name}</h4>'
        html_content += generate_aggregate_stats_table(stats)
        
        if score_key == 'sem_score_match_stats':
            score_data = [info[score_key]['mean'] for filename, info in model_data['scores'].items()]
        elif score_key == 'order_accuracy':
            score_data = [info[score_key]['normalized_kendall_tau'] for filename, info in model_data['scores'].items()]
        else:
            score_data = [info[score_key] for filename, info in model_data['scores'].items()]
        
        if score_data:
            score_svg = generate_histogram(score_data, chart_title, limits, max_frequencies, score_key)
            html_content += f'<div class="chart">{score_svg}</div>'
        
        html_content += '</div>'  # Close metric div
    
    html_content += f'<a href="#details-{model_data["model_id"]}">Jump to Model {model_data["model_id"]} Details</a>'
    html_content += '</div>'  # Close model-aggregate-stats div
    return html_content

def adjust_file_permissions(output_file):
    if os.getuid() == 0:
        host_uid, host_gid = 1000, 1000
        os.chown(output_file, host_uid, host_gid)
        print(f"File ownership changed for {output_file}")

def set_script_tag_values():
    script ="""
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
        });
    </script>
    """
    return script
    
def set_style_tag_values():
    style ="""
        <style>
            body { background-color: #263238; color: #ECEFF1; font-family: monospace; font-size: 1.1em; }
            pre { white-space: pre-wrap; font-size: smaller; }
            code { white-space: pre-wrap; background-color: black; padding: 4px; display: inline-block; }
            code.prompt { color: #80d4ff; background-color: transparent; min-width: 40ch; }
            code.response { color: #66ff66; width: 80ch; }
            code.gt { color: #ffa366; width: 80ch; }
            details summary { cursor: pointer; }
            table { width: 100%; border-collapse: collapse; }
            th, td { border: 1px solid #37474F; padding: 12px; text-align: left; font-size: 0.9em; overflow: visible; position: relative; }
            th.sortable:hover { cursor: pointer; }
        </style>
        """
    return style

def generate_html(data):
    html_content = '<html><head><title>Model Comparison Report</title>'
    html_content += set_script_tag_values()
    html_content += set_style_tag_values()
    html_content += '<h1>Model Comparison</h1>'
    
    metrics = [
        ('Final Score', 'final_score_stats', 'final_score', 'Final Score Distribution'),
        ('Sem Score Match Stats', 'matched_name_semantic_score_stats', 'sem_score_match_stats', 'Sem Score Match Stats Distribution'),
        ('Order Accuracy', 'order_accuracy_score_stats', 'order_accuracy', 'Order Accuracy Distribution'),
        ('Count Accuracy', 'count_accuracy_score_stats', 'count_accuracy', 'Count Accuracy Distribution')
    ]
    
    limits = compute_limits(data, metrics)
    max_frequencies = compute_max_frequencies(data, metrics)
    
    details_sections = ''
    
    for model_data in data:
        html_content += generate_model_overview(model_data, metrics, limits, max_frequencies)
        details_sections += generate_detailed_model_scores(model_data)

    html_content += details_sections
    html_content += '</body></html>'
    
    output_file = 'report.html'
    with open(output_file, 'w') as file:
        file.write(html_content)
    
    try:
        adjust_file_permissions(output_file)
    except Exception as e:
        print(f"Failed to set file level permissions for {output_file}: {e}")


if __name__ == "__main__":
    args = parse_arguments()
    data = load_data(args.scores_json_files)
    generate_html(data)
