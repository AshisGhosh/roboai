#!/usr/bin/python -u

import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel

from robosim.robosim import RoboSim

import logging
logging.basicConfig(level=logging.INFO)

# Create FastAPI instance
app = FastAPI()

robosim = RoboSim()

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

class Task(BaseModel):
    name: str
    type: str
    args: list | str

# Example route
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI! This is the robosim server."}

@app.on_event("startup")
async def startup_event():
    return robosim.setup()

@app.post("/test")
async def test():
    await add_task(Task(name="test_get_can", type="go_to_object", args="Can"))
    await robosim.start_execution()
    return {"msg": "Test task added and execution started."}

@app.post("/start")
async def start():
    return await robosim.start_async()

@app.get("/get_image")
async def get_image():
    logging.info("Getting image...")
    img = await robosim.get_image()
    logging.debug("Image received.")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    logging.debug("Image saved. Ready to stream.")
    return StreamingResponse(buf, media_type="image/png")

@app.post("/pause")
async def pause():
    return await robosim.pause_execution()

@app.post("/resume")
async def resume():
    return await robosim.resume_execution()

@app.post("/close")
async def close():
    return await robosim.close_renderer()

@app.post("/execute_tasks")
async def execute_tasks():
    return await robosim.execute_async()

@app.post("/add_task")
async def add_task(task: Task):
    logging.info(f"Adding task: {task.name} of type {task.type} with args {task.args}")
    try:
        robosim.add_task(task.name, task.type, task.args)
        return {"msg": "Task added"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_tasks")
async def get_tasks():
    tasks = [Task(name=t.name, type=t.function.__name__, args=t.args) for t in robosim.get_tasks()]
    return tasks


@app.get("/get_objects")
async def get_objects():
    return robosim.get_object_names()

    
    