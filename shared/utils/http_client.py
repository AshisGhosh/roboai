import os
from typing import Any, Dict, Optional

import asyncio
import httpx
from httpx import Timeout

from dotenv import load_dotenv
load_dotenv()

import logging
log = logging.getLogger("http_client")
log.setLevel(logging.DEBUG)

TIMEOUT_DEFAULT = 5.0

async def post_request(url: str, params: Optional[Dict[str, Any]] = None, data: Optional[Dict[str, Any]] = None, files: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, Any]] = None, timeout: float = TIMEOUT_DEFAULT) -> Dict[str, Any]:
    timeout = Timeout(timeout)
    log.debug(f"Sending POST request to {url}:")
    log.debug(f"    headers: {headers}")
    log.debug(f"    params: {params}")
    log.debug(f"    data: {data}")
    if files:
     log.debug(f"    files len: {len(files)}")
    log.debug(f"    timeout: {timeout}")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, params=params, json=data, files=files, headers=headers, timeout=timeout)
            response = response.json()
            log.debug(f"Response: {response}")
            return response
    except httpx.ReadTimeout as e:
        log.debug(
            f"Timeout sending POST request to {url} with params: {params} and timeout: {timeout}: {e}")
        return {"success": False, "text": f"httpx.ReadTimeout: Timeout sending POST request to {url} with params: {params} and timeout: {timeout}: {e}"}
    except Exception as e:
        log.debug(
            f"Error sending POST request to {url} with params: {params} and timeout: {timeout}: {e}")
        return {"success": False, "text": f"Error sending POST request to {url} with params: {params} and timeout: {timeout}: {e}"}


async def get_request(url: str) -> Dict[str, Any]:
    log.debug(f"Sending GET request to {url}")
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        log.debug(f"Response: {response.json()}")
        return response.json()
