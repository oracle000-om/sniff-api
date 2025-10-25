from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import torch
import torchvision.transforms as transforms
import timm
import io
import os
import json
import uuid
import cv2
from pathlib import Path
import hashlib
from datetime import datetime

from models.detection import PetDetector
from models.matching import PetMatcher
from models.embedding import PetEmbedder
from models.image_quality import ImageQualityChecker

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from PIL import Image


def log_event(event_type, data=None):
    """Simple event logging for analytics"""
    try:
        log_file = Path("data/analytics.json")
        log_file.parent.mkdir(exist_ok=True, parents=True)

        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "data": data or {},
        }

        if log_file.exists():
            with open(log_file, "r") as f:
                events = json.load(f)
        else:
            events = []

        events.append(event)

        # Keep last 10000 events
        with open(log_file, "w") as f:
            json.dump(events[-10000:], f, indent=2)
    except Exception as e:
        print(f"⚠️ Analytics logging failed: {e}")


images_dir = Path("data/images")
images_dir.mkdir(exist_ok=True, parents=True)

templates = Jinja2Templates(directory="templates")


app = FastAPI(
    title="Sniff API",
    description="Open-source pet recognition for reunification",
    version="1.0.0-mvp",
)

# Ensure data directory exists
os.makedirs("data/images", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Mount static files
app.mount("/data", StaticFiles(directory="data"), name="data")

# Connect to Milvus - use Lite on Railway, Docker locally
import os

RAILWAY_ENV = os.getenv("RAILWAY_ENVIRONMENT")
print(f"🔌 Connecting to Milvus ({'Lite' if RAILWAY_ENV else 'Docker'})...")

try:
    if RAILWAY_ENV:
        # Railway: Use Milvus Lite (embedded database)
        connections.connect(alias="default", uri="./milvus_demo.db")
    else:
        # Local: Use Docker Milvus
        connections.connect(alias="default", host="localhost", port="19530")

    print("✅ Connected to Milvus successfully")
except Exception as e:
    print(f"❌ Connection error: {e}")
    raise

# Define collection schema
COLLECTION_NAME = "pet_images"
DIM = 2048  # ResNet50 feature dimension

# Check if collection exists, create if not
if utility.has_collection(COLLECTION_NAME):
    try:
        collection = Collection(COLLECTION_NAME)
        collection.load()
        print(f"✅ Collection loaded. Total pets: {collection.num_entities}")
    except Exception as e:
        print(f"❌ ERROR: Collection exists but won't load: {e}")
        print("⚠️ MANUAL ACTION REQUIRED - Do not auto-delete production data!")
        raise  # Stop startup, don't delete data!

# Create collection if needed
if not utility.has_collection(COLLECTION_NAME):
    print(f"📦 Creating collection '{COLLECTION_NAME}'...")

    fields = [
        FieldSchema(
            name="pet_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True
        ),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
        FieldSchema(name="pet_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="species", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="microchip", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="location_found", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="finder_name", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="finder_contact", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="shelter_id", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="report_type", dtype=DataType.VARCHAR, max_length=50),
        FieldSchema(name="holding_pet", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="notes", dtype=DataType.VARCHAR, max_length=1000),
    ]

    schema = CollectionSchema(fields, description="Pet image recognition database")
    collection = Collection(COLLECTION_NAME, schema)

    # Create index for vector search
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index("embedding", index_params)
    collection.load()
    print("✅ Collection created, indexed, and loaded")

# Ensure collection is loaded
collection = Collection(COLLECTION_NAME)
collection.load()
print(f"✅ Collection ready. Total pets: {collection.num_entities}")

# Load AI model
print("🤖 Loading ResNet50 model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use torchvision ResNet50 instead of timm (more reliable, no auth needed)
try:
    from torchvision.models import resnet50, ResNet50_Weights

    # Load with pretrained weights
    weights = ResNet50_Weights.IMAGENET1K_V1
    model = resnet50(weights=weights)

    # Remove classification head (we only want features)
    model = torch.nn.Sequential(*list(model.children())[:-1])

    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded on {device}")

except Exception as e:
    print(f"⚠️ Torchvision model failed: {e}")
    print("Falling back to timm without pretrained weights...")

    # Fallback to timm without downloading
    model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded (random weights) on {device}")

# Image preprocessing (same as before)
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Image preprocessing
preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

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
def health_check():
    """Comprehensive health check for monitoring"""
    try:
        # Check Milvus connection
        collection = Collection(COLLECTION_NAME)
        collection.load()
        pet_count = collection.num_entities

        # Check disk space
        import shutil

        disk = shutil.disk_usage("/")
        disk_free_gb = disk.free / (1024**3)

        # Check image directory
        image_count = len(list(Path("data/images").glob("*")))

        # Check if images match database
        image_db_match = abs(image_count - pet_count) <= 5  # Allow small variance

        return {
            "status": "healthy",
            "database": "connected",
            "pets_registered": pet_count,
            "images_stored": image_count,
            "data_synced": image_db_match,
            "disk_free_gb": round(disk_free_gb, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }, 500


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
matcher = PetMatcher(collection_name="pet_images")
quality_checker = ImageQualityChecker()


@app.get("/api/v1/stats")
def get_stats():
    """Get statistics"""
    try:
        collection = Collection("pet_images")
        collection.load()
        count = collection.num_entities

        # Calculate total unique claimers across all pets
        claims_file = Path("data/claims.json")
        total_unique_claimers = 0
        total_claim_clicks = 0

        if claims_file.exists():
            with open(claims_file, "r") as f:
                claims_data = json.load(f)

            for pet_claims in claims_data.values():
                if isinstance(pet_claims, dict):
                    total_unique_claimers += len(pet_claims.get("ip_hashes", {}))
                    total_claim_clicks += pet_claims.get("total", 0)

        return {
            "status": "success",
            "total_pets": count,
            "total_claimers": total_unique_claimers,
            "total_claims": total_claim_clicks,
        }
    except Exception as e:
        print(f"Stats error: {e}")
        return {"status": "error", "total_pets": 0, "message": str(e)}


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
    holding_pet: str = Form("no"),
    microchip: str = Form(""),
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
        # Check image quality (warning only, never block)
        quality_result = quality_checker.check_quality(str(temp_path))
        quality_warning = None

        # Accept all images, warn about quality
        if not quality_result["is_good"]:
            quality_warning = "⚠️ Tip: Better lighting and focus improve match accuracy"

        # Detect and crop pet
        detections, cropped = detector.detect_and_crop(str(temp_path))

        # Error: No pets detected
        if len(cropped) == 0:
            temp_path.unlink()
            return {
                "status": "error",
                "message": "No pets detected in this image. Please upload a clearer photo with the pet clearly visible.",
            }

        # Warning: Multiple pets
        multiple_pets_warning = None
        if len(detections) > 1:
            multiple_pets_warning = f"Multiple pets detected ({len(detections)}). We registered the first one detected."

        # Generate embedding
        embedding = embedder.generate_body_embedding(cropped[0])

        # Generate pet ID
        pet_id = str(uuid.uuid4())

        # Save cropped image
        image_dir = Path("data/images")
        image_dir.mkdir(exist_ok=True, parents=True)
        image_path = image_dir / f"{pet_id}.jpg"
        cv2.imwrite(str(image_path), cropped[0])

        # Insert directly into Milvus (bypassing PetMatcher)
        collection = Collection(COLLECTION_NAME)

        # Prepare data in exact order matching schema
        insert_data = [
            [pet_id],  # 1. pet_id
            [embedding],  # 2. embedding
            [pet_name],  # 3. pet_name
            [
                species if species else detections[0].get("class", "unknown")
            ],  # 4. species
            [microchip],  # 5. microchip
            [location_found if report_type == "found_pet" else ""],  # 6. location_found
            [finder_name if report_type == "found_pet" else ""],  # 7. finder_name
            [finder_contact if report_type == "found_pet" else ""],  # 8. finder_contact
            [shelter_id if report_type == "shelter_intake" else ""],  # 9. shelter_id
            [report_type],  # 10. report_type
            [holding_pet],  # 11. holding_pet
            [notes],  # 12. notes
        ]

        # Insert and flush
        collection.insert(insert_data)
        collection.flush()

        # Clean up temp file
        temp_path.unlink()

        # Log success
        print(
            f"✅ Registered pet: {pet_name} ({species}) as {report_type} with ID: {pet_id}"
        )
        print(f"📊 Total pets in collection: {collection.num_entities}")

        # Build response
        response = {
            "status": "success",
            "pet_id": pet_id,
            "pet_name": pet_name,
            "species": species if species else detections[0].get("class", "unknown"),
            "report_type": report_type,
        }

        # Add warnings if any
        warnings = []
        if quality_warning:
            warnings.append(quality_warning)
        if multiple_pets_warning:
            warnings.append(multiple_pets_warning)

        if warnings:
            response["warning"] = " ".join(warnings)

        return response

    except Exception as e:
        print(f"❌ Registration error: {e}")
        import traceback

        traceback.print_exc()
        if temp_path.exists():
            temp_path.unlink()
        return {"status": "error", "message": str(e)}


@app.post("/api/v1/match")
async def match_pet(image: UploadFile = File(...)):
    """Find similar pets in the database"""

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
        multiple_pets_warning = None
        if len(detections) > 1:
            multiple_pets_warning = f"Multiple pets detected ({len(detections)}). Searching with the first one detected."

        # Generate embedding
        embedding = embedder.generate_body_embedding(cropped[0])

        # Search in Milvus directly (bypass matcher)
        collection = Collection(COLLECTION_NAME)
        collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[embedding],
            anns_field="embedding",
            param=search_params,
            limit=10,
            output_fields=[
                "pet_id",
                "pet_name",
                "species",
                "microchip",
                "location_found",
                "finder_name",
                "finder_contact",
                "shelter_id",
                "report_type",
            ],
        )

        # Process results
        matches = []
        for hits in results:
            for hit in hits:
                similarity_score = 1 / (1 + hit.distance)

                # Only include matches above 70% threshold
                if similarity_score > 0.70:
                    match_data = {
                        "pet_id": hit.entity.get("pet_id"),
                        "pet_name": hit.entity.get("pet_name"),
                        "species": hit.entity.get("species"),
                        "microchip": hit.entity.get("microchip"),
                        "location_found": hit.entity.get("location_found"),
                        "finder_name": hit.entity.get("finder_name"),
                        "finder_contact": hit.entity.get("finder_contact"),
                        "shelter_id": hit.entity.get("shelter_id"),
                        "report_type": hit.entity.get("report_type"),
                        "similarity_score": similarity_score,
                        "claims": 0,
                    }
                    matches.append(match_data)

        # Add claim counts to matches
        claims_file = Path("data/claims.json")
        if claims_file.exists():
            with open(claims_file, "r") as f:
                claims_data = json.load(f)

            for match in matches:
                pet_claims = claims_data.get(
                    match["pet_id"], {"total": 0, "ip_hashes": {}}
                )

                # IMPORTANT: Show unique claimers, not total clicks!
                if isinstance(pet_claims, dict):
                    # Count unique IP hashes (unique claimers)
                    unique_claimers = len(pet_claims.get("ip_hashes", {}))
                    match["claims"] = (
                        unique_claimers  # ← Changed from total to unique count
                    )
                    match["total_claims"] = pet_claims.get(
                        "total", 0
                    )  # Keep for analytics
                else:
                    # Old format fallback
                    match["claims"] = pet_claims
                    match["total_claims"] = pet_claims

        # Clean up temp file
        temp_path.unlink()

        print(f"🔍 Found {len(matches)} matches above 70% threshold")

        # Build response
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
        print(f"❌ Match error: {e}")
        import traceback

        traceback.print_exc()

        if temp_path.exists():
            temp_path.unlink()

        return {
            "status": "error",
            "message": f"Match search failed: {str(e)}",
            "matches": [],
        }


@app.post("/api/v1/claim")
async def claim_pet(request: Request):
    """
    Record a pet claim
    Rules:
    - Each IP can claim each pet ONCE
    - Each IP can claim up to 10 DIFFERENT pets total
    """
    try:
        body = await request.json()
        pet_id = body.get("pet_id")

        if not pet_id:
            return {"status": "error", "message": "Pet ID required"}

        # Get client IP and hash it for privacy (GDPR compliant)
        client_ip = request.client.host
        salt = "ZB5x0Wu2YDI0xj0ESuUeLKR0jHXnlMdQxNbCdTE73GA"
        ip_hash = hashlib.sha256(f"{client_ip}{salt}".encode()).hexdigest()[
            :16
        ]  # ← ADD THIS LINE

        # Load claims data
        claims_file = Path("data/claims.json")
        claims_file.parent.mkdir(exist_ok=True, parents=True)

        if claims_file.exists():
            with open(claims_file, "r") as f:
                claims_data = json.load(f)
        else:
            claims_data = {}

        # Initialize pet claims if needed
        if pet_id not in claims_data:
            claims_data[pet_id] = {
                "total": 0,
                "ip_hashes": [],  # List of IP hashes (each appears once)
                "timestamps": [],
            }

        # RULE 1: Check if this IP already claimed THIS specific pet
        if ip_hash in claims_data[pet_id]["ip_hashes"]:
            return {
                "status": "already_claimed",
                "message": "You have already claimed this pet.",
                "claims": len(claims_data[pet_id]["ip_hashes"]),
            }

        # RULE 2: Count how many DIFFERENT pets this IP has claimed total
        total_pets_claimed_by_this_ip = 0
        for pet_data in claims_data.values():
            if isinstance(pet_data, dict) and ip_hash in pet_data.get("ip_hashes", []):
                total_pets_claimed_by_this_ip += 1

        # Check if user has reached limit of 10 different pets
        MAX_PETS_PER_USER = 10

        if total_pets_claimed_by_this_ip >= MAX_PETS_PER_USER:
            return {
                "status": "limit_reached",
                "message": f"You have already claimed {MAX_PETS_PER_USER} different pets (the maximum allowed). If you need assistance, please contact the shelter directly.",
                "claims": len(claims_data[pet_id]["ip_hashes"]),
            }

        # Record the claim
        claims_data[pet_id]["ip_hashes"].append(ip_hash)
        claims_data[pet_id]["total"] = len(claims_data[pet_id]["ip_hashes"])
        claims_data[pet_id]["timestamps"].append(datetime.now().isoformat())

        # Save updated claims
        with open(claims_file, "w") as f:
            json.dump(claims_data, f, indent=2)

        # Calculate new totals
        new_total_claimed = total_pets_claimed_by_this_ip + 1
        remaining = MAX_PETS_PER_USER - new_total_claimed

        print(f"✅ Claim recorded: Pet {pet_id}, IP hash {ip_hash}")
        print(f"   This pet now has {claims_data[pet_id]['total']} unique claimers")
        print(
            f"   This user has claimed {new_total_claimed}/{MAX_PETS_PER_USER} different pets"
        )

        return {
            "status": "success",
            "message": "Claim recorded successfully! The shelter/finder will be notified.",
            "claims": claims_data[pet_id]["total"],
            "your_total_claims": new_total_claimed,
            "remaining_claims": remaining,
        }

    except Exception as e:
        print(f"❌ Claim error: {e}")
        import traceback

        traceback.print_exc()
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


@app.get("/api/v1/shelters")
def get_shelters():
    """Get list of registered shelters"""
    try:
        # Query all unique shelter IDs from registered pets
        results = matcher.collection.query(
            expr="report_type == 'shelter_intake'",
            output_fields=["shelter_id"],
            limit=1000,
        )

        # Get unique shelter names
        shelters = list(set([r["shelter_id"] for r in results if r.get("shelter_id")]))
        shelters.sort()

        return {"status": "success", "shelters": shelters}
    except Exception as e:
        return {"status": "error", "message": str(e), "shelters": []}
