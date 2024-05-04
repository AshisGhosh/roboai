import asyncio
from shared.utils.http_client import post_request

API_URL = "https://api-inference.huggingface.co/models/Efficient-Large-Model/VILA-2.7b"
headers = {"Authorization": "Bearer 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"}


async def _vila_query(text, image=None):
    json = {"inputs": text}
    response = await post_request(API_URL, headers=headers, data=json)
    print(response)
    return response


def vila_query(text, image=None):
    return asyncio.run(_vila_query(text, image))
