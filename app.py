from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
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

templates = Jinja2Templates(directory="templates")

# CORS middleware - allows requests from different origins (ports)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Accept from any origin (change in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],  # Allow any headers
)


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the UI"""
    return templates.TemplateResponse("index.html", {"request": request})


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


from models.matching import PetMatcher
from models.embedding import PetEmbedder

# Initialize (add after detector initialization)
embedder = PetEmbedder()
matcher = PetMatcher(collection_name="sniff_pets")


@app.post("/api/v1/register")
async def register_pet(
    image: UploadFile = File(...),
    pet_name: str = "Unknown",
    species: str = "unknown",
    shelter_id: str = "unknown",
    notes: str = "",
):
    """
    Register a new pet in the database

    Upload an image and metadata to add a pet to the matching database
    """
    # Save uploaded file temporarily
    temp_dir = Path("data/temp")
    temp_dir.mkdir(exist_ok=True)

    file_id = str(uuid.uuid4())
    temp_path = temp_dir / f"{file_id}_{image.filename}"

    with open(temp_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # Detect and crop
    detections, cropped = detector.detect_and_crop(str(temp_path))

    if len(cropped) == 0:
        temp_path.unlink()
        return {"status": "error", "message": "No pets detected in image"}

    # Generate embedding for first detected pet
    embedding = embedder.generate_body_embedding(cropped[0])

    # Register in database
    pet_id = matcher.register_pet(
        embedding=embedding,
        pet_name=pet_name,
        species=species if species != "unknown" else detections[0]["class"],
        shelter_id=shelter_id,
        notes=notes,
    )

    # Clean up temp file
    temp_path.unlink()

    return {
        "status": "success",
        "pet_id": pet_id,
        "pet_name": pet_name,
        "species": species,
        "detections": detections[0],
    }


@app.post("/api/v1/match")
async def match_pet(
    image: UploadFile = File(...), top_k: int = 5, threshold: float = 0.7
):
    """
    Find similar pets in the database

    Upload an image to search for matching pets
    """
    # Save uploaded file temporarily
    temp_dir = Path("data/temp")
    temp_dir.mkdir(exist_ok=True)

    file_id = str(uuid.uuid4())
    temp_path = temp_dir / f"{file_id}_{image.filename}"

    with open(temp_path, "wb") as f:
        content = await image.read()
        f.write(content)

    # Detect and crop
    detections, cropped = detector.detect_and_crop(str(temp_path))

    if len(cropped) == 0:
        temp_path.unlink()
        return {"status": "error", "message": "No pets detected in image"}

    # Generate embedding
    embedding = embedder.generate_body_embedding(cropped[0])

    # Search for matches
    matches = matcher.search_similar(
        embedding=embedding, top_k=top_k, threshold=threshold
    )

    # Clean up temp file
    temp_path.unlink()

    return {
        "status": "success",
        "query_detection": detections[0],
        "matches": matches,
        "count": len(matches),
    }


@app.get("/api/v1/stats")
def get_stats():
    """Get database statistics"""
    stats = matcher.get_stats()
    return stats


# Future endpoints will go here:
# @app.post("/api/v1/detect")
# @app.post("/api/v1/match")
# @app.post("/api/v1/register")
