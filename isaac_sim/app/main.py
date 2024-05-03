import nest_asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from roboai.robosim import SimManager
from roboai.standalone_stream_server import StreamServer

nest_asyncio.apply()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create FastAPI instance
app = FastAPI()

robosim = SimManager()
ss = StreamServer()

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
    return {"message": "Hello, FastAPI! This is the robosim server."}

@app.on_event("startup")
def startup_event():
    pass

@app.post("/test")
async def test():
    ss.start()
    return True

@app.post("/start_sim")
async def start_sim():
    # threading.Thread(target=robosim.start_sim).start()
    robosim.start_sim(headless=True)
    return True

@app.post("/run_sim")
async def run_sim():
    return robosim.run_sim()

@app.post("/close_sim")
async def close_sim():
    return await robosim.close_sim()