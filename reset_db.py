from pymilvus import connections, utility

# Connect to Milvus
connections.connect(alias="default", uri="./milvus_demo.db")

# Drop the collection
if utility.has_collection("sniff_pets"):
    utility.drop_collection("sniff_pets")
    print("Deleted collection: sniff_pets")
else:
    print("Collection sniff_pets does not exist")

print("Done! Restart your server to create the new collection.")
