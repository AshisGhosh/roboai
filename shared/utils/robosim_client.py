import asyncio
from shared.utils.http_client import get_request, post_request

SERVER_NAME = "http://localhost:8000"

async def _get_objects_on_table() -> list[str]:
    return await get_request(f"{SERVER_NAME}/get_objects")

def get_objects_on_table():
    return asyncio.run(_get_objects_on_table())

async def _pick(object_name: str):
    return await post_request(f"{SERVER_NAME}/add_task", data={"name": f"pick {object_name}", "type": "go_to_object", "args": object_name})

def pick(object_name: str):
    return asyncio.run(_pick(object_name))

async def _place(object_name: str):
    return await post_request(f"{SERVER_NAME}/add_task", data={"name": f"place {object_name}", "type": "go_to_object", "args": object_name})

def place(object_name: str):
    return asyncio.run(_place(object_name))