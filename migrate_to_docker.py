#!/usr/bin/env python3
"""Migrate from Milvus Lite to Docker Milvus"""

from pymilvus import (
    connections,
    Collection,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
)
from pathlib import Path
import json

# Constants
COLLECTION_NAME = "pet_images"
DIM = 2048


def export_data():
    """Export data from Milvus Lite"""
    print("📤 Exporting data from Milvus Lite...")

    # Connect to Lite
    connections.connect(alias="lite", uri="./milvus_demo.db")

    collection = Collection(COLLECTION_NAME, using="lite")
    collection.load()

    # Get all data
    results = collection.query(expr="pet_id != ''", output_fields=["*"], limit=10000)

    print(f"✅ Found {len(results)} pets to migrate")

    # Save to JSON
    with open("migration_data.json", "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Data exported to migration_data.json")

    connections.disconnect("lite")


def import_to_docker():
    """Import data to Docker Milvus"""
    print("\n📥 Importing data to Docker Milvus...")

    # Connect to Docker
    connections.connect(alias="docker", host="localhost", port="19530")

    # Load exported data
    with open("migration_data.json", "r") as f:
        data = json.load(f)

    if not data:
        print("⚠️ No data to import")
        connections.disconnect("docker")
        return

    print(f"📊 Found {len(data)} records in export file")

    # Debug: Check first record structure
    print(f"🔍 First record keys: {list(data[0].keys())}")
    print(f"🔍 Embedding type: {type(data[0]['embedding'])}")
    print(
        f"🔍 Embedding length: {len(data[0]['embedding']) if isinstance(data[0]['embedding'], list) else 'N/A'}"
    )

    # Drop existing collection if exists
    if utility.has_collection(COLLECTION_NAME, using="docker"):
        print(f"⚠️ Collection '{COLLECTION_NAME}' already exists, dropping...")
        utility.drop_collection(COLLECTION_NAME, using="docker")

    # Create schema
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

    schema = CollectionSchema(fields, description="Pet facial recognition database")
    collection = Collection(COLLECTION_NAME, schema, using="docker")

    print(f"✅ Collection '{COLLECTION_NAME}' created")

    # Prepare data for insertion - FIXED: Ensure embeddings are proper format
    print(f"📥 Preparing {len(data)} records for insertion...")

    # Extract and validate embeddings
    embeddings = []
    for item in data:
        emb = item["embedding"]
        # If embedding is nested list, flatten it
        if isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], list):
            emb = [val for sublist in emb for val in sublist]
        embeddings.append(emb)

    # Verify embedding dimensions
    print(f"🔍 First embedding length after processing: {len(embeddings[0])}")

    insert_data = [
        [item["pet_id"] for item in data],
        embeddings,  # Use processed embeddings
        [item["pet_name"] for item in data],
        [item["species"] for item in data],
        [item.get("microchip", "") for item in data],
        [item.get("location_found", "") for item in data],
        [item.get("finder_name", "") for item in data],
        [item.get("finder_contact", "") for item in data],
        [item.get("shelter_id", "") for item in data],
        [item["report_type"] for item in data],
        [item.get("holding_pet", "no") for item in data],
        [item.get("notes", "") for item in data],
    ]

    # Debug: Print counts
    print(f"🔍 Data structure lengths:")
    for i, field in enumerate(
        [
            "pet_id",
            "embedding",
            "pet_name",
            "species",
            "microchip",
            "location_found",
            "finder_name",
            "finder_contact",
            "shelter_id",
            "report_type",
            "holding_pet",
            "notes",
        ]
    ):
        print(f"   {field}: {len(insert_data[i])}")

    collection.insert(insert_data)
    collection.flush()

    print(f"✅ Inserted {len(data)} records")

    # Create index
    print("🔍 Creating search index...")
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
    }
    collection.create_index("embedding", index_params)

    # Load collection
    collection.load()

    # Verify
    final_count = collection.num_entities
    print(f"✅ Verified: Collection has {final_count} entities")

    connections.disconnect("docker")


if __name__ == "__main__":
    import sys

    if "--import" in sys.argv:
        print("🚀 Importing data to Docker Milvus...\n")
        import_to_docker()
        print("\n✅ Migration complete!")
    else:
        print("🚀 Starting migration from Milvus Lite to Docker...\n")
        export_data()
        print("\n⏸️  Now start Docker Milvus with: docker-compose up -d")
        print("⏸️  Wait 60 seconds for it to start")
        print("⏸️  Then run: python3 migrate_to_docker.py --import")
