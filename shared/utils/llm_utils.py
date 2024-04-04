from litellm import embedding
import numpy as np

from dotenv import load_dotenv
load_dotenv("shared/.env")  # take environment variables from .env.

import logging
log = logging.getLogger("llm_utils")
log.setLevel(logging.INFO)

def log_debug(msg):
    log.debug(msg)
    # print(msg)

def log_info(msg):
    log.info(msg)
    # print(msg)

def get_embedding(text):
    log_debug(f"Getting embedding for text: {text}")
    response = embedding(
        model='huggingface/mixedbread-ai/mxbai-embed-large-v1', 
        input=[text]
    )
    log_debug(f"Embedding received: {response}")
    return response["data"][0]["embedding"]

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)

def get_closest_text(text: str, text_list: list[str], k: int = 1) -> str:
    query_vector = get_embedding(text)
    vectors = [get_embedding(text) for text in text_list]
    similarities = [cosine_similarity(query_vector, vector) for vector in vectors]
    if k > 1:
        closest_indices = np.argsort(similarities)[-k:]
        return [text_list[i] for i in closest_indices]
    closest_index = np.argmax(similarities)
    return text_list[closest_index]
