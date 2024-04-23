# README


## Launch Docker
`docker compose up isaac-sim`
Enter docker:
`docker exec -it roboai-isaac-sim-1 bash`

## Run pthon standalone (will launch sim as well)
 `./python.sh roboai/test.py`

## Run jupyter
** requires local Nucleus server **
https://docs.omniverse.nvidia.com/nucleus/latest/workstation.html 

`./jupyter_notebook.sh --allow-root roboai/test_nb.`


## Isaac Slow Loading Issue (v2023.1.0)
https://github.com/NVIDIA-Omniverse/OmniIsaacGymEnvs/issues/92#issuecomment-1797057491

```
def check_server(server: str, path: str) -> bool:
    """Check a specific server for a path

    Args:
        server (str): Name of Nucleus server
        path (str): Path to search

    Returns:
        bool: True if folder is found
    """
    carb.log_info("Checking path: {}{}".format(server, path))
    # Increase hang detection timeout
    if "localhost" not in server:
        omni.client.set_hang_detection_time_ms(10000)
        result, _ = omni.client.stat("{}{}".format(server, path))
        if result == Result.OK:
            carb.log_info("Success: {}{}".format(server, path))
            return True
    carb.log_info("Failure: {}{} not accessible".format(server, path))
    return False
```

to:

```
def check_server(server: str, path: str, timeout: float = 10.0) -> bool:
    """Check a specific server for a path

    Args:
        server (str): Name of Nucleus server
        path (str): Path to search
        timeout (float): Default value: 10 seconds

    Returns:
        bool: True if folder is found
    """
    carb.log_info("Checking path: {}{}".format(server, path))
    # Increase hang detection timeout
    if "localhost" not in server:
        omni.client.set_hang_detection_time_ms(20000)
        result, _ = omni.client.stat("{}{}".format(server, path))
        if result == Result.OK:
            carb.log_info("Success: {}{}".format(server, path))
            return True
    carb.log_info("Failure: {}{} not accessible".format(server, path))
    return False
```