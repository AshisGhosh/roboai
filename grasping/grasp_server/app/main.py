#!/usr/bin/python -u

import io
import base64
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from grasp_server.grasp_det_seg import GraspServer

import logging

logging.basicConfig(level=logging.INFO)

# Create GraspServer instance
grasp = GraspServer()

# Create FastAPI instance
app = FastAPI()

# List of allowed origins (you can use '*' to allow all origins)
origins = [
    "http://localhost:3000",  # Allow your Next.js app
    # Add any other origins as needed
]

# Add CORSMiddleware to the application
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


# Example route
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI! This is the grasp server."}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read the image file
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)

    # Decode the image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection
    result, img = grasp.detect(image)
    # Return the result
    return JSONResponse(content={"result": result, "image": get_image_response(img)})


@app.post("/test")
async def test():
    result, img = grasp.test_detect()
    return JSONResponse(content={"result": result, "image": get_image_response(img)})


def get_image_response(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    buf.seek(0)
    base64_image = base64.b64encode(buf.getvalue()).decode("utf-8")
    return base64_image
    # return StreamingResponse(buf, media_type="image/jpeg")
