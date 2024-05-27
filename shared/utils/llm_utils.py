import litellm
import ollama
import logging
import numpy as np
from fastembed import TextEmbedding
import asyncio

from shared.utils.model_server_client import _embed

from dotenv import load_dotenv

load_dotenv("shared/.env")  # take environment variables from .env.

log = logging.getLogger("llm_utils")
log.setLevel(logging.INFO)


def log_debug(msg):
    log.debug(msg)
    # print(msg)


def log_info(msg):
    log.info(msg)
    # print(msg)


async def get_embedding_sentence_transformers(text):
    log_debug(f"Getting sentence_transformer/HF embedding for text: {text}")
    response = await _embed(text)
    return response["embedding"]


def get_embedding_ollama(text):
    log_debug(f"Getting ollama embedding for text: {text}")
    response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
    return response["embedding"]


def get_embedding_litellm(text):
    log_debug(f"Getting litellm/HF embedding for text: {text}")
    response = litellm.embedding(
        model="huggingface/mixedbread-ai/mxbai-embed-large-v1", input=[text]
    )
    log_debug(f"Embedding received: {response}")
    return response["data"][0]["embedding"]


global fast_embed_model
fast_embed_model = None


def get_embedding_fastembed(text):
    global fast_embed_model
    if not fast_embed_model:
        fast_embed_model = TextEmbedding("mixedbread-ai/mxbai-embed-large-v1")
    embed = list(fast_embed_model.embed(text))[0]
    return embed


async def get_embedding(text):
    log_debug(f"Getting embedding for text: {text}")
    return get_embedding_fastembed(text)


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    return dot_product / (norm_v1 * norm_v2)


async def get_closest_text(
    text: str, text_list: list[str], k: int = 1, threshold: float = 0.5
) -> str:
    log_info(f"Getting closest text for: '{text}' in list: {text_list}")
    query_vector = await get_embedding(text)
    log_debug(f"Query vector: {query_vector}")
    vectors = [await get_embedding(text) for text in text_list]
    similarities = [cosine_similarity(query_vector, vector) for vector in vectors]
    log_debug(f"Similarities: {similarities}")
    if k > 1:
        closest_indices = np.argsort(similarities)[-k:]
        log_info(f"Closest texts: {[text_list[i] for i in closest_indices]}")
        return [text_list[i] for i in closest_indices]
    closest_index = np.argmax(similarities)
    if similarities[closest_index] < threshold:
        log_info(f"Similarity below threshold: {similarities[closest_index]}")
        return None
    log_info(f"Closest text: {text_list[closest_index]}")
    return text_list[closest_index]


def get_closest_text_sync(
    text: str, text_list: list[str], k: int = 1, threshold: float = 0.5
):
    return asyncio.run(get_closest_text(text, text_list, k, threshold))


async def get_most_important(texts: list[str] | str, k: int = 1):
    log_info(f"Getting most important text from: {texts}")
    if isinstance(texts, list):
        texts = " ".join(texts)

    texts_embedding = await get_embedding(texts)

    texts = texts.split()
    vectors = [await get_embedding(text) for text in texts]

    similarities = [cosine_similarity(texts_embedding, vector) for vector in vectors]
    log_debug(f"Similarities: {similarities}")
    closest_indices = np.argsort(similarities)[-k:]
    log_info(f"Closest texts: {[texts[i] for i in closest_indices]}")
    return [texts[i] for i in closest_indices]


def get_most_important_sync(texts: list[str], k: int = 1):
    return asyncio.run(get_most_important(texts, k))
