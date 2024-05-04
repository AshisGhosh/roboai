from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from robotic_grasping_server.robotic_grasping import GraspGenerator

import logging

log = logging.getLogger("robotic_grasping_server app")
log.setLevel(logging.INFO)

app = FastAPI()
grasp = GraspGenerator(visualize=True)


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


@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI! This is the robotic grasping server."}


@app.on_event("startup")
async def startup_event():
    log.info("Starting up the grasp server...")
    grasp.load_model()


@app.post("/get_grasps")
async def get_grasps(
    rgb_image: UploadFile = File(...), depth_image: UploadFile = File(...)
):
    log.debug("Received get_grasp request.")
    rgb_image = Image.open(rgb_image.file)
    depth_image = Image.open(depth_image.file)
    return grasp.run(rgb_image, depth_image)
