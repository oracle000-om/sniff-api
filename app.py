from starlette.requests import Request
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import UploadFile, File
from models.detection import PetDetector
from models.image_quality import ImageQualityChecker
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

# Initialize models
detector = PetDetector()
embedder = PetEmbedder()
matcher = PetMatcher(collection_name="sniff_pets")
quality_checker = ImageQualityChecker()


@app.post("/api/v1/register")
async def register_pet(
    image: UploadFile = File(...),
    pet_name: str = Form("Unknown"),
    species: str = Form("unknown"),
    shelter_id: str = Form("unknown"),
    notes: str = Form(""),
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

    try:
        # Check image quality
        quality_result = quality_checker.check_quality(str(temp_path))

        # Store quality warnings (don't block registration)
        quality_warnings = []
        if not quality_result["is_good"]:
            quality_warnings = quality_result["warnings"]

        # Detect and crop
        detections, cropped = detector.detect_and_crop(str(temp_path))

        # Error: No pets detected
        if len(cropped) == 0:
            temp_path.unlink()
            return {
                "status": "error",
                "message": "No pets detected in this image. Please upload a clearer photo with the pet clearly visible.",
            }

        # Warning: Multiple pets detected (use first one)
        if len(detections) > 1:
            multiple_pets_warning = f"Multiple pets detected ({len(detections)}). We registered the first one detected. To register the other pet(s), please upload a separate image with just that pet visible."
        else:
            multiple_pets_warning = None

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

        response = {
            "status": "success",
            "pet_id": pet_id,
            "pet_name": pet_name,
            "species": species if species != "unknown" else detections[0]["class"],
            "detections": detections[0],
            "confidence": detections[0]["confidence"],
        }

        if multiple_pets_warning:
            response["warning"] = multiple_pets_warning
        elif quality_warnings:  # Only show if no multiple pets warning
            response["warning"] = " ".join(quality_warnings)

        return response

        if multiple_pets_warning:
            response["warning"] = multiple_pets_warning

        return response

    except Exception as e:
        # Clean up temp file if it still exists
        if temp_path.exists():
            temp_path.unlink()

        return {"status": "error", "message": f"Registration failed: {str(e)}"}


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

    try:
        # Detect and crop
        detections, cropped = detector.detect_and_crop(str(temp_path))

        # Error: No pets detected
        if len(cropped) == 0:
            temp_path.unlink()
            return {
                "status": "error",
                "message": "No pets detected in this image. Please upload a clearer photo with the pet clearly visible.",
            }

        # Warning: Multiple pets detected
        if len(detections) > 1:
            multiple_pets_warning = f"Multiple pets detected ({len(detections)}). We registered the first one detected. To register the other pet(s), please upload a separate image with just that pet visible."
        else:
            multiple_pets_warning = None

        # Generate embedding
        embedding = embedder.generate_body_embedding(cropped[0])

        # Search for matches
        matches = matcher.search_similar(
            embedding=embedding, top_k=top_k, threshold=threshold
        )

        # Clean up temp file
        temp_path.unlink()

        response = {
            "status": "success",
            "query_detection": detections[0],
            "matches": matches,
            "count": len(matches),
        }

        if multiple_pets_warning:
            response["warning"] = multiple_pets_warning

        return response

    except Exception as e:
        # Clean up temp file if it still exists
        if temp_path.exists():
            temp_path.unlink()

        return {"status": "error", "message": f"Match search failed: {str(e)}"}


@app.get("/api/v1/stats")
def get_stats():
    """Get database statistics"""
    stats = matcher.get_stats()
    return stats


# Future endpoints will go here:
# @app.post("/api/v1/detect")
# @app.post("/api/v1/match")
# @app.post("/api/v1/register")
