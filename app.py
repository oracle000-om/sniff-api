from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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


# Future endpoints will go here:
# @app.post("/api/v1/detect")
# @app.post("/api/v1/match")
# @app.post("/api/v1/register")
