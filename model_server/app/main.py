#!/usr/bin/python -u
import io
from PIL import Image

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from fastapi.responses import JSONResponse

from model_server.hf_moondream2 import HuggingFaceMoonDream2

moondream = HuggingFaceMoonDream2()


import logging
logging.basicConfig(level=logging.INFO)


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
    return {"message": "Hello, FastAPI! This is the model server."}


@app.post("/answer_question")
async def answer_question(file: UploadFile = File(...), question: str = ""):
    # Read the image file
    image_bytes = await file.read()
    # Convert bytes to a file-like object
    image_stream = io.BytesIO(image_bytes)

    # Use PIL to open the image
    image = Image.open(image_stream)
    # Perform object detection
    result = moondream.answer_question_from_image(image, question)
    # Return the result
    return JSONResponse(content={"result": result})