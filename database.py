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
        try:
            uri = (
                "./milvus_demo.db" if os.getenv("RAILWAY_ENVIRONMENT") else "localhost"
            )
            connections.connect(alias="default", host=uri, port="19530")

            if not utility.has_collection(COLLECTION_NAME):
                self._create_collection()

            self.collection = Collection(COLLECTION_NAME)
            self.collection.load()
            self.available = True
            print("✅ Milvus Connected")
        except Exception as e:
            print(f"❌ Milvus Offline: {e}")

    def _create_collection(self):
        fields = [
            FieldSchema(
                name="pet_id", dtype=DataType.VARCHAR, max_length=100, is_primary=True
            ),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIM),
            FieldSchema(name="pet_name", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="species", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="report_type", dtype=DataType.VARCHAR, max_length=50),
            # ... add other fields from your original code here
        ]
        schema = CollectionSchema(fields, "Pet DB")
        col = Collection(COLLECTION_NAME, schema)
        col.create_index(
            "embedding",
            {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
        )


db = DBManager()
