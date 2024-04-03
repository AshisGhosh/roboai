import asyncio
import base64
import io
import numpy as np
from PIL import Image
import cv2
from shared.utils.grasp_client import _check_server, _get_grasp_from_image

class GraspHandler:
    def __init__(self, robot):
        self.robot = robot

    async def get_grasp_from_image(self, image: Image):
        res = await _get_grasp_from_image(image)
        self.show_image(res["image"])
        return res["result"]
    
    async def check_server(self):
        return await _check_server()
    
    def show_image(self, base64_image):
        image_bytes = base64.b64decode(base64_image)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Display the image using OpenCV
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

