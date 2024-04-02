#!/usr/bin/python -u

import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse

from grasp_server.grasp import GraspServer

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
    image = await file.read()
    # Perform object detection
    result = grasp.detect(image)
    # Return the result
    return result

@app.post("/test")
async def test():
    result = grasp.test_detect()
    return result
