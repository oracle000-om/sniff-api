import os
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)

COLLECTION_NAME = "pet_images"
DIM = 2048


class DBManager:
    def __init__(self):
        self.available = False
        self.collection = None
        self.connect()

    def connect(self):
        """Connect to Milvus (local or Zilliz Cloud)"""
        try:
            host = os.getenv("MILVUS_HOST", "localhost")
            port = os.getenv("MILVUS_PORT", "19530")
            token = os.getenv("MILVUS_TOKEN", "")

            # Zilliz Cloud connection (with token)
            if token:
                # Serverless uses URI without port
                connections.connect(alias="default", uri=f"https://{host}", token=token)
                print(f"🔗 Connecting to Zilliz Serverless: {host}")
            # Local Milvus connection
            else:
                connections.connect(alias="default", host=host, port=port)
                print(f"🔗 Connecting to local Milvus: {host}:{port}")

            # Create collection if it doesn't exist
            if not utility.has_collection(COLLECTION_NAME):
                self._create_collection()

            # Load collection
            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
            self.available = True
            print("✅ Milvus Connected")

        except Exception as e:
            self.available = False
            print(f"❌ Milvus Offline: {e}")

    def _create_collection(self):
        """Create the pet_images collection with schema"""
        fields = [
            FieldSchema(
                name="pet_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="pet_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="species", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="report_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="shelter_id", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="microchip", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="finder_name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="finder_contact", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="location_found", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="holding_pet", dtype=DataType.VARCHAR, max_length=10),
            FieldSchema(name="notes", dtype=DataType.VARCHAR, max_length=1000),
        ]

        schema = CollectionSchema(fields, "Pet facial recognition database")
        col = Collection(COLLECTION_NAME, schema)

        # Create index for vector search
        col.create_index(
            "embedding",
            {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")


# Global database instance
db = DBManager()
