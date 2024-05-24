import httpx
import logging

from typing import Any, Dict, Optional

from httpx import Timeout

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("http_client")
log.setLevel(logging.DEBUG)

TIMEOUT_DEFAULT = 5.0


async def post_request(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, Any]] = None,
    timeout: float = TIMEOUT_DEFAULT,
) -> Dict[str, Any]:
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
            response = await client.post(
                url,
                params=params,
                json=data,
                files=files,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code == 200:
                response = response.json()
                log.debug(f"Response: {response}")
                return response
            raise Exception(f"Error:{response.status_code}: {response.text}")
    except httpx.ReadTimeout as e:
        log.debug(
            f"Timeout sending POST request to {url} with params: {params} and timeout: {timeout}: {e}"
        )
        return {
            "success": False,
            "text": f"httpx.ReadTimeout: Timeout sending POST request to {url} with params: {params} and timeout: {timeout}: {e}",
        }
    except Exception as e:
        log.debug(
            f"Error sending POST request to {url} with params: {params} and timeout: {timeout}: {e}"
        )
        return {
            "success": False,
            "text": f"Error sending POST request to {url} with params: {params} and timeout: {timeout}: {e}",
        }


async def get_request(url: str, timeout:float = TIMEOUT_DEFAULT) -> Dict[str, Any]:
    log.debug(f"Sending GET request to {url}")
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, timeout=timeout)
        log.debug(response)
        if response.status_code == 200:
            log.debug(f"Response: {response.json()}")
            return response.json()
        else:
            log.debug(f"Error: {response.text}")
            return {"success": False, "text": f"{response.status}{response.text}"}


async def get_image_request(url: str, timeout:float = TIMEOUT_DEFAULT) -> bytes:
    log.debug(f"Sending GET request to {url}")
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url, timeout=timeout)
        log.debug(response)
        if response.status_code == 200:
            image_data = bytearray()
            for chunk in response.iter_bytes():
                image_data += chunk
            log.debug(f"Response: ({type(image_data)}) {image_data[:10]}")
            return image_data
        else:
            log.debug(f"Error: {response.text}")
            return b""
