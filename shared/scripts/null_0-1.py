import json
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def store_words_in_dict(file_paths):
    words_dict = {}
    idx = 0
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            data = json.load(f)
            for item in data:
                for conversation in item['conversations']:
                    for word in re.findall(r'\b\w+\b', conversation['value'].lower()):
                        words_dict[idx] = word
                        idx += 1
    return words_dict

def generate_synthetic_data(words_dict, mean_length, std_dev_length, num_phrases=4, seed=None):
    if seed is not None:
        np.random.seed(seed)

    synthetic_data = {
        "objects": {
            "count": num_phrases,
            "names": []
        }
    }

    dict_size = len(words_dict)
    for _ in range(num_phrases):
        word_count = max(1, int(np.random.normal(mean_length, std_dev_length)))
        indices = np.random.choice(dict_size, word_count)
        phrase = ' '.join([words_dict[i] for i in indices])
        synthetic_data["objects"]["names"].append(phrase)

    return synthetic_data

if __name__ == '__main__':
    training_file_paths = ['blip_laion_cc_sbu_558k.json', 'llava_v1_5_mix665k.json']
    words_dict = store_words_in_dict(training_file_paths)

    mean_length = 2.880674448767834
    std_dev_length = 1.3040006531267216

    num_generations = 500
    num_phrases_per_generation = 4

    np.random.seed(27)  # Set main seed

    print(f"Processing {num_generations} synthetic responses. Please wait...")

    synthetic_responses = {}
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_synthetic_data, words_dict, mean_length, std_dev_length, num_phrases_per_generation, hash(i + 27) % (2**32)) for i in range(num_generations)]

        for index, future in enumerate(futures, start=1):
            synthetic_responses[str(index)] = future.result()

    with open('synthetic_responses.json', 'w') as f:
        json.dump({"synthetic_responses": synthetic_responses}, f, indent=4)
    print(f'Successfully written {num_generations} synthetic responses to synthetic_responses.json')
