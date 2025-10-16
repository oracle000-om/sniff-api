from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from models.detection import PetDetector
import uuid
from pathlib import Path

app = FastAPI(
    title="Sniff API",
    description="Open-source pet recognition for reunification",
    version="1.0.0-mvp",
)

# CORS middleware - allows requests from different origins (ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept from any origin (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # Allow any headers
)


@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Sniff API is ALIVE!",  # Changed this
        "version": "1.0.0-mvp",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Health check endpoint - used by monitoring tools"""
    return {"status": "healthy"}


# Initialize detector
detector = PetDetector()


@app.post("/api/v1/detect")
async def detect_pet(image: UploadFile = File(...)):
    """
    Detect pets in uploaded image

    Returns detection results with bounding boxes
    """
    # Save uploaded file temporarily
    temp_dir = Path("data/temp")
    temp_dir.mkdir(exist_ok=True)

    file_id = str(uuid.uuid4())
    temp_path = temp_dir / f"{file_id}_{image.filename}"

    with open(temp_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # Run detection
    detections, _ = detector.detect_and_crop(str(temp_path))

    # Clean up temp file
    temp_path.unlink()

    return {"detection_id": file_id, "detections": detections, "count": len(detections)}


# Future endpoints will go here:
# @app.post("/api/v1/detect")
# @app.post("/api/v1/match")
# @app.post("/api/v1/register")
