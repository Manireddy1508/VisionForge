import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import CLIPProcessor, CLIPModel
from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility,
)

# Initialize CLIP model for image embeddings
CLIP_MODEL = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
model = CLIPModel.from_pretrained(CLIP_MODEL)

# Milvus collection name
COLLECTION_NAME = "image_generations"


def connect_to_milvus():
    """Connect to Milvus server."""
    try:
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
        )
        print("✅ Connected to Milvus server")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {str(e)}")
        raise


def create_collection():
    """Create Milvus collection if it doesn't exist."""
    try:
        if utility.has_collection(COLLECTION_NAME):
            print(f"Collection {COLLECTION_NAME} already exists")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="input_prompt", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(
                name="enhanced_prompt", dtype=DataType.VARCHAR, max_length=2000
            ),
            FieldSchema(name="edited_prompt", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(
                name="input_image_path", dtype=DataType.VARCHAR, max_length=500
            ),
            FieldSchema(
                name="output_image_path", dtype=DataType.VARCHAR, max_length=500
            ),
            FieldSchema(name="model_used", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
            FieldSchema(name="image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
        ]

        schema = CollectionSchema(fields=fields, description="Image generation records")
        collection = Collection(name=COLLECTION_NAME, schema=schema)

        # Create IVF_FLAT index for image embeddings
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024},
        }
        collection.create_index(field_name="image_embedding", index_params=index_params)
        print(f"✅ Created collection {COLLECTION_NAME} with index")
    except Exception as e:
        print(f"❌ Failed to create collection: {str(e)}")
        raise


def generate_image_embedding(image_path: str) -> np.ndarray:
    """Generate CLIP embedding for an image."""
    try:
        # Handle both local files and URLs
        if image_path.startswith(("http://", "https://")):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)

        # Preprocess image
        inputs = processor(images=image, return_tensors="pt", padding=True)

        # Generate embedding
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            embedding = image_features.numpy()[0]

        return embedding
    except Exception as e:
        print(f"❌ Failed to generate image embedding: {str(e)}")
        raise


def insert_full_generation_record(
    input_prompt: str,
    output_image_path: str,
    model_used: str,
    enhanced_prompt: Optional[str] = None,
    edited_prompt: Optional[str] = None,
    input_image_path: Optional[str] = None,
    category: Optional[str] = None,
) -> int:
    """Insert a complete generation record into Milvus."""
    try:
        # Generate embedding for output image
        image_embedding = generate_image_embedding(output_image_path)

        # Prepare data for insertion
        data = {
            "input_prompt": input_prompt,
            "enhanced_prompt": enhanced_prompt or "",
            "edited_prompt": edited_prompt or "",
            "input_image_path": input_image_path or "",
            "output_image_path": output_image_path,
            "model_used": model_used,
            "category": category or "",
            "timestamp": int(time.time()),
            "image_embedding": image_embedding.tolist(),
        }

        # Insert into Milvus
        collection = Collection(COLLECTION_NAME)
        collection.insert([data])
        collection.flush()

        # Get the inserted ID
        results = collection.query(
            expr=f'input_prompt == "{input_prompt}" and timestamp == {data["timestamp"]}',
            output_fields=["id"],
        )
        if results:
            return results[0]["id"]
        return -1
    except Exception as e:
        print(f"❌ Failed to insert generation record: {str(e)}")
        raise


def search_similar_images(
    image_path: str, top_k: int = 5, category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search for similar images in the database."""
    try:
        # Generate query embedding
        query_embedding = generate_image_embedding(image_path)

        # Prepare search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # Build search expression
        expr = None
        if category:
            expr = f'category == "{category}"'

        # Search in Milvus
        collection = Collection(COLLECTION_NAME)
        collection.load()
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="image_embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "input_prompt",
                "enhanced_prompt",
                "edited_prompt",
                "input_image_path",
                "output_image_path",
                "model_used",
                "category",
                "timestamp",
            ],
        )

        # Format results
        similar_images = []
        for hits in results:
            for hit in hits:
                similar_images.append(
                    {
                        "id": hit.id,
                        "distance": hit.distance,
                        "input_prompt": hit.entity.get("input_prompt"),
                        "enhanced_prompt": hit.entity.get("enhanced_prompt"),
                        "edited_prompt": hit.entity.get("edited_prompt"),
                        "input_image_path": hit.entity.get("input_image_path"),
                        "output_image_path": hit.entity.get("output_image_path"),
                        "model_used": hit.entity.get("model_used"),
                        "category": hit.entity.get("category"),
                        "timestamp": hit.entity.get("timestamp"),
                    }
                )

        return similar_images
    except Exception as e:
        print(f"❌ Failed to search similar images: {str(e)}")
        raise


def get_record_by_id(record_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a generation record by its ID."""
    try:
        collection = Collection(COLLECTION_NAME)
        results = collection.query(
            expr=f"id == {record_id}",
            output_fields=[
                "input_prompt",
                "enhanced_prompt",
                "edited_prompt",
                "input_image_path",
                "output_image_path",
                "model_used",
                "category",
                "timestamp",
            ],
        )
        return results[0] if results else None
    except Exception as e:
        print(f"❌ Failed to get record by ID: {str(e)}")
        raise


def _check_for_duplicate(
    input_prompt: str, output_image_path: str, threshold: float = 0.95
) -> Optional[int]:
    """Check if a similar generation already exists."""
    try:
        # Generate embedding for output image
        image_embedding = generate_image_embedding(output_image_path)

        # Search for similar images
        collection = Collection(COLLECTION_NAME)
        collection.load()
        results = collection.search(
            data=[image_embedding.tolist()],
            anns_field="image_embedding",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=1,
            output_fields=["id", "input_prompt"],
        )

        # Check if any result is similar enough
        for hits in results:
            for hit in hits:
                if hit.distance < (1 - threshold):
                    return hit.id
        return None
    except Exception as e:
        print(f"❌ Failed to check for duplicate: {str(e)}")
        raise


# Initialize Milvus connection and collection on import
try:
    connect_to_milvus()
    create_collection()
except Exception as e:
    print(f"⚠️ Failed to initialize Milvus: {str(e)}")
