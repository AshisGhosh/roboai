# Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

from omni.isaac.kit import SimulationApp


class StreamServer:
    def __init__(self):
        pass

    def start(self):
        # This sample enables a livestream server to connect to when running headless
        CONFIG = {
            "width": 1280,
            "height": 720,
            "window_width": 1920,
            "window_height": 1080,
            "headless": True,
            "renderer": "RayTracedLighting",
            "display_options": 3286,  # Set display options to show default grid
        }

        # Start the omniverse application
        kit = SimulationApp(launch_config=CONFIG)

        from omni.isaac.core.utils.extensions import enable_extension

        # Default Livestream settings
        kit.set_setting("/app/window/drawMouse", True)
        kit.set_setting("/app/livestream/proto", "ws")
        kit.set_setting("/app/livestream/websocket/framerate_limit", 120)
        kit.set_setting("/ngx/enabled", False)

        # Note: Only one livestream extension can be enabled at a time
        # Enable Native Livestream extension
        # Default App: Streaming Client from the Omniverse Launcher
        enable_extension("omni.kit.livestream.native")

        # Enable WebSocket Livestream extension(Deprecated)
        # Default URL: http://localhost:8211/streaming/client/
        # enable_extension("omni.services.streamclient.websocket")

        # Enable WebRTC Livestream extension
        # Default URL: http://localhost:8211/streaming/webrtc-client/
        # enable_extension("omni.services.streamclient.webrtc")

        # Run until closed
        while kit._app.is_running() and not kit.is_exiting():
            # Run in realtime mode, we don't specify the step size
            kit.update()

        kit.close()
