"""
Matching module - vector search for similar pets using Milvus
"""

import numpy as np
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)
from typing import List, Dict, Tuple
import uuid


class PetMatcher:
    """Handles pet registration and matching using Milvus vector database"""

    def __init__(self, collection_name: str = "pet_embeddings"):
        """
        Initialize Milvus connection and collection

        Args:
            collection_name: Name of the Milvus collection
        """
        self.collection_name = collection_name

        # Connect to Milvus Lite (local embedded)
        print("Connecting to Milvus...")
        connections.connect(
            alias="default", uri="./milvus_demo.db"  # Local file for Milvus Lite
        )

        # Create or load collection
        self._setup_collection()

        print(f"Milvus collection '{collection_name}' ready")

    def _setup_collection(self):
        """Create collection if it doesn't exist"""

        # Check if collection exists
        if utility.has_collection(self.collection_name):
            print(f"Loading existing collection: {self.collection_name}")
            self.collection = Collection(self.collection_name)
            self.collection.load()
            return

        print(f"Creating new collection: {self.collection_name}")

        # Define schema
        fields = [
            FieldSchema(
                name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=2048),
            FieldSchema(name="pet_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="species", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(
                name="report_type", dtype=DataType.VARCHAR, max_length=20
            ),  # "shelter_intake" or "found_pet"
            FieldSchema(name="shelter_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="finder_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="finder_contact", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="location_found", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="microchip", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="notes", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="image_url", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="claims", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(
            fields=fields, description="Pet embeddings for matching"
        )

        # Create collection
        self.collection = Collection(name=self.collection_name, schema=schema)

        # Create index for fast search
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
        self.collection.create_index(field_name="embedding", index_params=index_params)
        self.collection.load()

    def register_pet(
        self,
        embedding: np.ndarray,
        pet_name: str,
        species: str,
        report_type: str = "shelter_intake",
        shelter_id: str = "unknown",
        finder_name: str = "",
        finder_contact: str = "",
        location_found: str = "",
        microchip: str = "",
        notes: str = "",
        image_url: str = "",
    ) -> str:
        """
        Register a new pet in the database

        Args:
            embedding: 2048-d feature vector
            pet_name: Name of the pet
            species: 'dog' or 'cat'
            report_type: 'shetler_intake' or 'found_pet'
            shelter_id: Shelter identifier (for shelter intakes)
            finder_name: Name of the person who found the pet (for found pets)
            finder_contact: Contact info of the finder (for found pets)
            location_found: Location where the pet was found (for found pets)
            notes: Additional notes

        Returns:
            pet_id: Unique identifier for the pet
        """
        pet_id = str(uuid.uuid4())

        # Prepare data
        data = [
            [pet_id],
            [embedding.tolist()],
            [pet_name],
            [species],
            [report_type],
            [shelter_id],
            [finder_name],
            [finder_contact],
            [location_found],
            [microchip],
            [notes],
            [image_url],
            [0],
        ]

        # Insert into Milvus
        self.collection.insert(data)
        self.collection.flush()

        print(
            f"Registered pet: {pet_name} ({species}) as {report_type} with ID: {pet_id}"
        )
        return pet_id

    def search_similar(
        self, embedding: np.ndarray, top_k: int = 5, threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for similar pets

        Args:
            embedding: Query embedding (2048-d)
            top_k: Number of results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of matches with metadata and scores
        """
        # Search parameters
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}

        # Search
        results = self.collection.search(
            data=[embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=[
                "pet_name",
                "species",
                "report_type",
                "shelter_id",
                "finder_name",
                "finder_contact",
                "location_found",
                "microchip",
                "notes",
                "image_url",
            ],
        )

        # Format results
        matches = []
        for hits in results:
            for hit in hits:
                similarity = float(hit.distance)

                # Filter by threshold
                if similarity >= threshold:
                    match = {
                        "pet_id": hit.id,
                        "similarity_score": similarity,
                        "pet_name": hit.entity.get("pet_name"),
                        "species": hit.entity.get("species"),
                        "report_type": hit.entity.get("report_type"),
                        "shelter_id": hit.entity.get("shelter_id"),
                        "finder_name": hit.entity.get("finder_name"),
                        "finder_contact": hit.entity.get("finder_contact"),
                        "location_found": hit.entity.get("location_found"),
                        "microchip": hit.entity.get("microchip"),
                        "notes": hit.entity.get("notes"),
                        "image_url": hit.entity.get("image_url"),
                    }
                    matches.append(match)

        return matches

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        self.collection.flush()
        num_entities = self.collection.num_entities

        return {
            "collection_name": self.collection_name,
            "total_pets": num_entities,
            "index_type": "IVF_FLAT",
            "metric": "COSINE",
        }

    def delete_pet(self, pet_id: str):
        """Delete a pet from the database"""
        expr = f'id == "{pet_id}"'
        self.collection.delete(expr)
        self.collection.flush()
        print(f"Deleted pet: {pet_id}")


# Test function
def test_matcher():
    """Test the matching system"""
    from models.detection import PetDetector
    from models.embedding import PetEmbedder

    print("\n=== Testing Pet Matcher ===\n")

    # Initialize
    detector = PetDetector()
    embedder = PetEmbedder()
    matcher = PetMatcher(collection_name="test_pets")

    # Test image
    test_image = "data/test_images/dog.jpg"

    print(f"Processing: {test_image}")

    # Detect, crop, embed
    detections, cropped = detector.detect_and_crop(test_image)

    if len(cropped) == 0:
        print("No pets detected!")
        return

    # Generate embedding
    embedding = embedder.generate_body_embedding(cropped[0])

    # Register pet
    pet_id = matcher.register_pet(
        embedding=embedding,
        pet_name="Test Dog",
        species="dog",
        shelter_id="test_shelter",
        notes="Golden retriever test",
    )

    # Search for similar pets (should find itself)
    print("\n=== Searching for similar pets ===")
    matches = matcher.search_similar(embedding, top_k=3, threshold=0.7)

    print(f"Found {len(matches)} match(es):")
    for i, match in enumerate(matches):
        print(f"\n  Match {i+1}:")
        print(f"    Name: {match['pet_name']}")
        print(f"    Species: {match['species']}")
        print(f"    Similarity: {match['similarity_score']:.4f}")
        print(f"    Shelter: {match['shelter_id']}")
        print(f"    ID: {match['pet_id']}")

    # Stats
    print("\n=== Database Stats ===")
    stats = matcher.get_stats()
    print(f"Total pets registered: {stats['total_pets']}")

    # Cleanup
    print(f"\nCleaning up test pet...")
    matcher.delete_pet(pet_id)
    print("Test complete!")


if __name__ == "__main__":
    test_matcher()
