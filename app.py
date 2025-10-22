from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
import cv2
import json

from models.detection import PetDetector
from models.matching import PetMatcher
from models.embedding import PetEmbedder
from models.image_quality import ImageQualityChecker

images_dir = Path("data/images")
images_dir.mkdir(exist_ok=True, parents=True)

templates = Jinja2Templates(directory="templates")


app = FastAPI(
    title="Sniff API",
    description="Open-source pet recognition for reunification",
    version="1.0.0-mvp",
)

app.mount("/data/images", StaticFiles(directory="data/images"), name="images")

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


@app.get("/api/v1/stats")
async def get_stats():
    try:
        # Use matcher's get_stats method
        stats = matcher.get_stats()

        return {"status": "success", "total_pets": stats["total_pets"]}
    except Exception as e:
        return {"status": "error", "message": str(e), "total_pets": 0}


@app.post("/api/v1/register")
async def register_pet(
    image: UploadFile = File(...),
    pet_name: str = Form("Unknown"),
    species: str = Form("unknown"),
    report_type: str = Form("shelter_intake"),
    shelter_id: str = Form(""),
    finder_name: str = Form(""),
    finder_contact: str = Form(""),
    location_found: str = Form(""),
    microchip: str = Form(""),  # ADD THIS LINE
    notes: str = Form(""),
):
    """Register a new pet (shelter intake or found pet report)"""

    # Save uploaded file temporarily
    temp_dir = Path("data/temp")
    temp_dir.mkdir(exist_ok=True, parents=True)

    file_id = str(uuid.uuid4())
    temp_path = temp_dir / f"{file_id}_{image.filename}"

    with open(temp_path, "wb") as f:
        content = await image.read()
        f.write(content)

    try:
        # Check image quality (warning only, don't block)
        quality_result = quality_checker.check_quality(str(temp_path))
        quality_warning = None

        if not quality_result["is_good"]:
            quality_warning = "⚠️ Image quality: " + " ".join(quality_result["warnings"])

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
        multiple_pets_warning = None
        if len(detections) > 1:
            multiple_pets_warning = f"Multiple pets detected ({len(detections)}). We registered the first one detected."

        # Generate embedding
        embedding = embedder.generate_body_embedding(cropped[0])

        # Register in database with appropriate fields
        pet_id = matcher.register_pet(
            embedding=embedding,
            pet_name=pet_name,
            species=species if species != "unknown" else detections[0]["class"],
            report_type=report_type,
            shelter_id=shelter_id if report_type == "shelter_intake" else "",
            finder_name=finder_name if report_type == "found_pet" else "",
            finder_contact=finder_contact if report_type == "found_pet" else "",
            location_found=location_found if report_type == "found_pet" else "",
            microchip=microchip,  # ADD THIS LINE
            notes=notes,
        )

        # Save the cropped image before cleanup
        image_dir = Path("data/images")
        image_dir.mkdir(exist_ok=True, parents=True)
        image_filename = f"{pet_id}.jpg"
        image_path = image_dir / image_filename
        cv2.imwrite(str(image_path), cropped[0])

        # Clean up temp file
        temp_path.unlink()

        # Build response with warnings
        response = {
            "status": "success",
            "pet_id": pet_id,
            "pet_name": pet_name,
            "species": species,
            "report_type": report_type,
        }

        # Combine warnings if any exist
        warnings = []
        if quality_warning:
            warnings.append(quality_warning)
        if multiple_pets_warning:
            warnings.append(multiple_pets_warning)

        if warnings:
            response["warning"] = " ".join(warnings)

        return response

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        return {"status": "error", "message": str(e)}


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

        # Add claim counts to matches
        import json

        claims_file = Path("data/claims.json")
        if claims_file.exists():
            with open(claims_file, "r") as f:
                claims_data = json.load(f)
        else:
            claims_data = {}

        for match in matches:
            pet_claims = claims_data.get(match["pet_id"], {"total": 0, "ips": []})
            # Handle both old format (int) and new format (dict)
            if isinstance(pet_claims, dict):
                match["claims"] = pet_claims.get("total", 0)
            else:
                # Old format: just an integer
                match["claims"] = pet_claims

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


@app.post("/api/v1/pets/{pet_id}/claim")
def claim_pet(pet_id: str, request: Request):
    """
    Increment claim count for a pet and track by IP
    """
    try:
        # Get client IP
        client_ip = request.client.host

        # Track claims in a simple JSON file
        claims_file = Path("data/claims.json")
        claims_file.parent.mkdir(exist_ok=True, parents=True)

        import json

        # Load existing claims
        if claims_file.exists():
            with open(claims_file, "r") as f:
                claims_data = json.load(f)
        else:
            claims_data = {}

        # Structure: {"pet_id": {"total": 3, "ips": ["ip1", "ip2", "ip3"]}}
        if pet_id not in claims_data:
            claims_data[pet_id] = {"total": 0, "ips": []}

        # Check if this IP already claimed
        if client_ip in claims_data[pet_id]["ips"]:
            return {
                "status": "error",
                "message": "You have already claimed this pet. Your claim has been recorded in the system.",  # UPDATED
                "claims": claims_data[pet_id]["total"],
            }

        # Add new claim
        claims_data[pet_id]["ips"].append(client_ip)
        claims_data[pet_id]["total"] += 1

        # Save updated claims
        with open(claims_file, "w") as f:
            json.dump(claims_data, f, indent=2)

        return {
            "status": "success",
            "message": "Claim recorded successfully!",  # UPDATED
            "claims": claims_data[pet_id]["total"],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/v1/pets/{pet_id}")
def delete_pet(pet_id: str):
    """
    Delete a pet from the database

    Args:
        pet_id: The pet's unique identifier
    """
    try:
        matcher.delete_pet(pet_id)
        return {"status": "success", "message": f"Pet {pet_id} deleted successfully"}
    except Exception as e:
        return {"status": "error", "message": f"Failed to delete pet: {str(e)}"}


# Future endpoints will go here:
# @app.post("/api/v1/detect")
# @app.post("/api/v1/match")
# @app.post("/api/v1/register")
