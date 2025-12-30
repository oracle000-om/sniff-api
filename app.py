from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import uuid
import cv2
import os
import json
import hashlib
from datetime import datetime

from database import db
from models.detection import PetDetector
from models.embedding import PetEmbedder
from models.image_quality import ImageQualityChecker

app = FastAPI(
    title="Sniff API",
    description="Open-source pet recognition for reunification",
    version="1.0.0-mvp",
)

# Ensure directories exist
os.makedirs("data/images", exist_ok=True)
os.makedirs("data/temp", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Mount static files
app.mount("/data", StaticFiles(directory="data"), name="data")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI models once
detector = PetDetector()
embedder = PetEmbedder()
quality_checker = ImageQualityChecker()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/ways-to-help", response_class=HTMLResponse)
async def ways_to_help(request: Request):
    """Serve the ways to help page"""
    return templates.TemplateResponse("ways-to-help.html", {"request": request})


@app.get("/say-hi", response_class=HTMLResponse)
async def say_hi(request: Request):
    """Serve the say hi page"""
    return templates.TemplateResponse("say-hi.html", {"request": request})


@app.get("/health")
def health_check():
    """Comprehensive health check for monitoring"""
    try:
        if not db.available:
            return {
                "status": "degraded",
                "database": "offline",
                "timestamp": datetime.now().isoformat(),
            }

        pet_count = db.collection.num_entities if db.collection else 0

        # Check disk space
        import shutil

        disk = shutil.disk_usage("/")
        disk_free_gb = disk.free / (1024**3)

        # Check image directory
        image_count = len(list(Path("data/images").glob("*")))

        return {
            "status": "healthy",
            "database": "connected",
            "pets_registered": pet_count,
            "images_stored": image_count,
            "disk_free_gb": round(disk_free_gb, 2),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
        }


@app.get("/api/v1/stats")
def get_stats():
    """Get statistics"""
    try:
        if not db.available:
            return {"status": "error", "total_pets": 0, "message": "Database offline"}

        count = db.collection.num_entities if db.collection else 0

        # Calculate total unique claimers
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

    if not db.available:
        raise HTTPException(
            status_code=503, detail="Database offline - please try again later"
        )

    # Save uploaded file temporarily
    temp_dir = Path("data/temp")
    temp_dir.mkdir(exist_ok=True, parents=True)

    file_id = str(uuid.uuid4())
    temp_path = temp_dir / f"{file_id}_{image.filename}"

    with open(temp_path, "wb") as f:
        content = await image.read()
        f.write(content)

    try:
        # Check image quality (warning only)
        quality_result = quality_checker.check_quality(str(temp_path))
        quality_warning = None

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

        # Insert into database
        insert_data = [
            [pet_id],
            [embedding],
            [pet_name],
            [species if species else detections[0].get("class", "unknown")],
            [microchip],
            [location_found if report_type == "found_pet" else ""],
            [finder_name if report_type == "found_pet" else ""],
            [finder_contact if report_type == "found_pet" else ""],
            [shelter_id if report_type == "shelter_intake" else ""],
            [report_type],
            [holding_pet],
            [notes],
        ]

        db.collection.insert(insert_data)
        db.collection.flush()

        # Clean up temp file
        temp_path.unlink()

        print(
            f"✅ Registered pet: {pet_name} ({species}) as {report_type} with ID: {pet_id}"
        )
        print(f"📊 Total pets in collection: {db.collection.num_entities}")

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

    if not db.available:
        raise HTTPException(
            status_code=503, detail="Database offline - please try again later"
        )

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

        # Search in database
        db.collection.load()

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = db.collection.search(
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

                if isinstance(pet_claims, dict):
                    unique_claimers = len(pet_claims.get("ip_hashes", {}))
                    match["claims"] = unique_claimers
                    match["total_claims"] = pet_claims.get("total", 0)
                else:
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
    """Record a pet claim"""
    try:
        body = await request.json()
        pet_id = body.get("pet_id")

        if not pet_id:
            return {"status": "error", "message": "Pet ID required"}

        # Get client IP and hash it for privacy
        client_ip = request.client.host
        salt = "ZB5x0Wu2YDI0xj0ESuUeLKR0jHXnlMdQxNbCdTE73GA"
        ip_hash = hashlib.sha256(f"{client_ip}{salt}".encode()).hexdigest()[:16]

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
                "ip_hashes": [],
                "timestamps": [],
            }

        # Check if this IP already claimed THIS pet
        if ip_hash in claims_data[pet_id]["ip_hashes"]:
            return {
                "status": "already_claimed",
                "message": "You have already claimed this pet.",
                "claims": len(claims_data[pet_id]["ip_hashes"]),
            }

        # Count how many DIFFERENT pets this IP has claimed
        total_pets_claimed_by_this_ip = 0
        for pet_data in claims_data.values():
            if isinstance(pet_data, dict) and ip_hash in pet_data.get("ip_hashes", []):
                total_pets_claimed_by_this_ip += 1

        MAX_PETS_PER_USER = 10

        if total_pets_claimed_by_this_ip >= MAX_PETS_PER_USER:
            return {
                "status": "limit_reached",
                "message": f"You have already claimed {MAX_PETS_PER_USER} different pets (the maximum allowed).",
                "claims": len(claims_data[pet_id]["ip_hashes"]),
            }

        # Record the claim
        claims_data[pet_id]["ip_hashes"].append(ip_hash)
        claims_data[pet_id]["total"] = len(claims_data[pet_id]["ip_hashes"])
        claims_data[pet_id]["timestamps"].append(datetime.now().isoformat())

        # Save updated claims
        with open(claims_file, "w") as f:
            json.dump(claims_data, f, indent=2)

        new_total_claimed = total_pets_claimed_by_this_ip + 1
        remaining = MAX_PETS_PER_USER - new_total_claimed

        print(f"✅ Claim recorded: Pet {pet_id}, IP hash {ip_hash}")

        return {
            "status": "success",
            "message": "Claim recorded successfully!",
            "claims": claims_data[pet_id]["total"],
            "your_total_claims": new_total_claimed,
            "remaining_claims": remaining,
        }

    except Exception as e:
        print(f"❌ Claim error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/api/v1/shelters")
def get_shelters():
    """Get list of registered shelters"""
    try:
        if not db.available:
            return {"status": "error", "message": "Database offline", "shelters": []}

        results = db.collection.query(
            expr="report_type == 'shelter_intake'",
            output_fields=["shelter_id"],
            limit=1000,
        )

        shelters = list(set([r["shelter_id"] for r in results if r.get("shelter_id")]))
        shelters.sort()

        return {"status": "success", "shelters": shelters}
    except Exception as e:
        return {"status": "error", "message": str(e), "shelters": []}
