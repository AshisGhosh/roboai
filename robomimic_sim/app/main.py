#!/usr/bin/python -u

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
import io

from robomimic_sim.robomimic_sim import RobomimicSim

# Create FastAPI instance
app = FastAPI()

robosim = RobomimicSim()

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
    return {"message": "Hello, FastAPI! This is the robomimic server."}

@app.on_event("startup")
async def startup_event():
    return robosim.setup()

@app.post("/run")
async def run():
    print("Running robomimic simulation...")
    return await robosim.start_rollout()

@app.post("/reset")
async def reset():
    print("Resetting robomimic simulation...")
    return await robosim.reset()

@app.post("/start_renderer")
async def start_renderer():
    print("Starting robomimic simulation...")
    return await robosim.start_renderer()

@app.post("/close_renderer")
async def close_renderer():
    print("Closing robomimic simulation...")
    return await robosim.close_renderer()

@app.get("/get_policy")
async def get_policy():
    return repr(robosim.policy)

@app.get("/get_image")
async def get_image():
    print("Getting image...")
    img = robosim.get_image()
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")

