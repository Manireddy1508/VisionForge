import os
import time
from typing import List, Dict, Any, Optional, Union
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
from uuid import uuid4
from pathlib import Path
import logging
import json

# Initialize CLIP model for image embeddings
CLIP_MODEL = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
model = CLIPModel.from_pretrained(CLIP_MODEL)

# Milvus collection names
PROMPT_COLLECTION = "prompt_embeddings"
INPUT_IMAGE_COLLECTION = "input_image_embeddings"
OUTPUT_IMAGE_COLLECTION = "output_image_embeddings"

# Directory for storing input images
INPUT_IMAGES_DIR = Path("input_images")
INPUT_IMAGES_DIR.mkdir(exist_ok=True)

# Logging setup
logging.basicConfig(
    filename='milvus_operations.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def log_operation(operation: str, details: dict):
    """Log Milvus operations with details."""
    logging.info(f"{operation}: {json.dumps(details)}")

def connect_to_milvus():
    """Connect to Milvus server."""
    try:
        # Check if already connected
        if connections.has_connection("default"):
            print("✅ Already connected to Milvus server")
            return True
            
        # Try to connect
        connections.connect(
            alias="default",
            host=os.getenv("MILVUS_HOST", "localhost"),
            port=os.getenv("MILVUS_PORT", "19530"),
        )
        print("✅ Connected to Milvus server")
        return True
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {str(e)}")
        return False


def create_collection():
    """Create the Milvus collections if they don't exist."""
    max_retries = 3
    retry_delay = 2  # seconds

    for attempt in range(max_retries):
        try:
            # Check if collections exist
            if (utility.has_collection(PROMPT_COLLECTION) and 
                utility.has_collection(INPUT_IMAGE_COLLECTION) and 
                utility.has_collection(OUTPUT_IMAGE_COLLECTION)):
                print(f"✅ Collections already exist")
                return True

            # Define prompt embeddings collection schema
            prompt_fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=36),
                FieldSchema(name="user_prompt", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="enhanced_prompt", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="final_prompt", dtype=DataType.VARCHAR, max_length=1000),
                FieldSchema(name="prompt_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
                FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)  # JSON string
            ]

            # Define input image embeddings collection schema
            input_image_fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=36),
                FieldSchema(name="input_image_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="input_image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
                FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)  # JSON string
            ]

            # Define output image embeddings collection schema
            output_image_fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=36),
                FieldSchema(name="output_image_path", dtype=DataType.VARCHAR, max_length=500),
                FieldSchema(name="output_image_embedding", dtype=DataType.FLOAT_VECTOR, dim=512),
                FieldSchema(name="model_used", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="timestamp", dtype=DataType.INT64),
                FieldSchema(name="status", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=50),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=2000)  # JSON string
            ]

            # Create collections one by one with individual error handling
            collections_to_create = [
                (PROMPT_COLLECTION, prompt_fields, "Prompt embeddings and metadata", "prompt_embedding"),
                (INPUT_IMAGE_COLLECTION, input_image_fields, "Input image embeddings and metadata", "input_image_embedding"),
                (OUTPUT_IMAGE_COLLECTION, output_image_fields, "Output image embeddings and metadata", "output_image_embedding")
            ]

            for collection_name, fields, description, embedding_field in collections_to_create:
                try:
                    schema = CollectionSchema(fields=fields, description=description, enable_dynamic_field=True)
                    collection = Collection(name=collection_name, schema=schema)
                    
                    # Create vector index
                    collection.create_index(
                        field_name=embedding_field,
                        index_params={
                            "metric_type": "L2",
                            "index_type": "IVF_FLAT",
                            "params": {"nlist": 1024}
                        }
                    )
                    
                    # Add Trie index on session_id for faster lookups
                    collection.create_index(
                        field_name="session_id",
                        index_params={
                            "index_type": "Trie"
                        }
                    )
                    
                    print(f"✅ Created collection {collection_name} with schema and dynamic fields enabled")
                except Exception as e:
                    print(f"❌ Failed to create collection {collection_name}: {str(e)}")
                    raise

            return True

        except Exception as e:
            print(f"❌ Attempt {attempt + 1}/{max_retries} failed to create collections: {str(e)}")
            if attempt < max_retries - 1:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("❌ All attempts to create collections failed")
                return False


def ensure_collections():
    """Ensure collections exist before any operation."""
    if not utility.has_collection(PROMPT_COLLECTION) or not utility.has_collection(INPUT_IMAGE_COLLECTION) or not utility.has_collection(OUTPUT_IMAGE_COLLECTION):
        return create_collection()
    return True


def save_input_image(image: Union[str, Image.Image, bytes]) -> str:
    """Save an input image to the input_images directory and return its path."""
    try:
        # Ensure input directory exists
        INPUT_IMAGES_DIR.mkdir(exist_ok=True, parents=True)
        
        # Generate unique filename
        image_id = str(uuid4())
        image_path = INPUT_IMAGES_DIR / f"{image_id}.png"
        
        # Handle different input types
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                response = requests.get(image)
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image)
        elif isinstance(image, bytes):
            img = Image.open(BytesIO(image))
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
            
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        # Save image
        img.save(image_path, format='PNG')
        print(f"✅ Saved input image to {image_path}")
        return str(image_path)
    except Exception as e:
        print(f"❌ Failed to save input image: {str(e)}")
        print(f"Image type: {type(image)}")
        if isinstance(image, Image.Image):
            print(f"Image mode: {image.mode}")
            print(f"Image size: {image.size}")
        raise


def generate_image_embedding(image_input: Union[str, Image.Image]) -> np.ndarray:
    """Generate CLIP embedding for an image."""
    try:
        # Handle both PIL Image objects and file paths
        if isinstance(image_input, str):
            if image_input.startswith(("http://", "https://")):
                response = requests.get(image_input)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_input)
        else:
            image = image_input

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


def generate_text_embedding(text: str) -> np.ndarray:
    """Generate embedding for text using CLIP model."""
    try:
        # Truncate text to fit within model's max length (77 tokens)
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
        return text_features.numpy()[0]
    except Exception as e:
        print(f"❌ Failed to generate text embedding: {str(e)}")
        return None


def ensure_collections_loaded():
    """Ensure collections exist and are loaded before any operation."""
    try:
        if not ensure_collections():
            return False
            
        # Load all collections
        for collection_name in [PROMPT_COLLECTION, INPUT_IMAGE_COLLECTION, OUTPUT_IMAGE_COLLECTION]:
            if utility.has_collection(collection_name):
                collection = Collection(collection_name)
                collection.load()
        return True
    except Exception as e:
        print(f"❌ Failed to load collections: {str(e)}")
        return False


def insert_prompt_record(
    session_id: str,
    user_prompt: str,
    enhanced_prompt: Optional[str] = None,
    final_prompt: Optional[str] = None,
    prompt_embedding: Optional[np.ndarray] = None,
    status: str = "pending",
    version: str = "OpenAI-v1",
    metadata: Optional[dict] = None
) -> int:
    """Insert a prompt record into the prompt_embeddings collection."""
    try:
        # Ensure collections exist and are loaded
        if not ensure_collections_loaded():
            raise Exception("Failed to ensure collections are loaded")

        # Generate prompt embedding if not provided
        if prompt_embedding is None:
            prompt_embedding = generate_text_embedding(final_prompt or user_prompt)

        # Prepare data for insertion
        data = {
            "session_id": session_id,
            "user_prompt": user_prompt,
            "enhanced_prompt": enhanced_prompt or "",
            "final_prompt": final_prompt or user_prompt,
            "prompt_embedding": prompt_embedding.tolist(),
            "timestamp": int(time.time()),
            "status": status,
            "version": version,
            "metadata": json.dumps(metadata) if metadata else "{}"
        }

        # Insert into Milvus
        collection = Collection(PROMPT_COLLECTION)
        collection.insert([data])
        collection.flush()

        # Log the operation
        log_operation("INSERT_PROMPT", {
            "session_id": session_id,
            "status": status,
            "version": version,
            "timestamp": data["timestamp"]
        })

        # Get the inserted ID
        results = collection.query(
            expr=f'session_id == "{session_id}"',
            output_fields=["id"],
        )
        if results:
            return results[0]["id"]
        return -1
    except Exception as e:
        print(f"❌ Failed to insert prompt record: {str(e)}")
        raise


def insert_image_record(
    session_id: str,
    input_image_path: Optional[str] = None,
    output_image_path: Optional[str] = None,
    input_image_embedding: Optional[np.ndarray] = None,
    output_image_embedding: Optional[np.ndarray] = None,
    model_used: str = "OpenAI",
    category: Optional[str] = None,
    status: str = "pending",
    version: str = "OpenAI-v1",
    metadata: Optional[dict] = None
) -> int:
    """Insert image records into the appropriate collections."""
    try:
        # Ensure collections exist and are loaded
        if not ensure_collections_loaded():
            raise Exception("Failed to ensure collections are loaded")

        # Insert input image record if provided
        if input_image_path and input_image_embedding is not None:
            input_data = {
                "session_id": session_id,
                "input_image_path": input_image_path,
                "input_image_embedding": input_image_embedding.tolist(),
                "timestamp": int(time.time()),
                "status": status,
                "version": version,
                "metadata": json.dumps(metadata) if metadata else "{}"
            }
            input_collection = Collection(INPUT_IMAGE_COLLECTION)
            input_collection.insert([input_data])
            input_collection.flush()

            # Log the operation
            log_operation("INSERT_INPUT_IMAGE", {
                "session_id": session_id,
                "status": status,
                "version": version,
                "timestamp": input_data["timestamp"]
            })

        # Insert output image record if provided
        if output_image_path and output_image_embedding is not None:
            output_data = {
                "session_id": session_id,
                "output_image_path": output_image_path,
                "output_image_embedding": output_image_embedding.tolist(),
                "model_used": model_used,
                "category": category or "",
                "timestamp": int(time.time()),
                "status": status,
                "version": version,
                "metadata": json.dumps(metadata) if metadata else "{}"
            }
            output_collection = Collection(OUTPUT_IMAGE_COLLECTION)
            output_collection.insert([output_data])
            output_collection.flush()

            # Log the operation
            log_operation("INSERT_OUTPUT_IMAGE", {
                "session_id": session_id,
                "status": status,
                "version": version,
                "model_used": model_used,
                "timestamp": output_data["timestamp"]
            })

        return 0
    except Exception as e:
        print(f"❌ Failed to insert image record: {str(e)}")
        raise


def search_similar_prompts(
    prompt: str,
    top_k: int = 5,
    category: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Search for similar prompts in the database."""
    try:
        # Generate query embedding
        query_embedding = generate_text_embedding(prompt)

        # Prepare search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

        # Build search expression
        expr = None
        if category:
            expr = f'category == "{category}"'

        # Search in Milvus
        collection = Collection(PROMPT_COLLECTION)
        collection.load()
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="prompt_embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "session_id",
                "user_prompt",
                "enhanced_prompt",
                "final_prompt",
                "timestamp"
            ],
        )

        # Format results
        similar_prompts = []
        for hits in results:
            for hit in hits:
                similar_prompts.append({
                    "id": hit.id,
                    "distance": float(hit.distance),
                    "session_id": hit.entity.get("session_id"),
                    "user_prompt": hit.entity.get("user_prompt"),
                    "enhanced_prompt": hit.entity.get("enhanced_prompt"),
                    "final_prompt": hit.entity.get("final_prompt"),
                    "timestamp": hit.entity.get("timestamp")
                })

        return similar_prompts
    except Exception as e:
        print(f"❌ Failed to search similar prompts: {str(e)}")
        raise


def search_similar_images(
    image_path: str,
    top_k: int = 5,
    category: Optional[str] = None,
    search_input_images: bool = False
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

        # Determine which field to search
        anns_field = "input_image_embedding" if search_input_images else "output_image_embedding"

        # Search in Milvus
        collection = Collection(INPUT_IMAGE_COLLECTION)
        collection.load()
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field=anns_field,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "session_id",
                "input_image_path",
                "model_used",
                "category",
                "timestamp"
            ],
        )

        # Format results
        similar_images = []
        for hits in results:
            for hit in hits:
                similar_images.append({
                    "id": hit.id,
                    "distance": float(hit.distance),
                    "session_id": hit.entity.get("session_id"),
                    "input_image_path": hit.entity.get("input_image_path"),
                    "model_used": hit.entity.get("model_used"),
                    "category": hit.entity.get("category"),
                    "timestamp": hit.entity.get("timestamp")
                })

        return similar_images
    except Exception as e:
        print(f"❌ Failed to search similar images: {str(e)}")
        raise


def search_similar_images_by_embedding(
    embedding: np.ndarray,
    top_k: int = 5,
    category: Optional[str] = None,
    search_input_images: bool = False
) -> List[Dict[str, Any]]:
    """Search for similar images in the database using a provided embedding."""
    try:
        # Prepare search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        expr = f'category == "{category}"' if category else None
        anns_field = "input_image_embedding" if search_input_images else "output_image_embedding"
        collection = Collection(INPUT_IMAGE_COLLECTION if search_input_images else OUTPUT_IMAGE_COLLECTION)
        collection.load()
        results = collection.search(
            data=[embedding.tolist()],
            anns_field=anns_field,
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=[
                "session_id",
                "input_image_path" if search_input_images else "output_image_path",
                "model_used" if not search_input_images else None,
                "category",
                "timestamp"
            ],
        )
        similar_images = []
        for hits in results:
            for hit in hits:
                similar_images.append({
                    "id": hit.id,
                    "distance": float(hit.distance),
                    "session_id": hit.entity.get("session_id"),
                    "input_image_path": hit.entity.get("input_image_path") if search_input_images else None,
                    "output_image_path": hit.entity.get("output_image_path") if not search_input_images else None,
                    "model_used": hit.entity.get("model_used") if not search_input_images else None,
                    "category": hit.entity.get("category"),
                    "timestamp": hit.entity.get("timestamp")
                })
        return similar_images
    except Exception as e:
        print(f"❌ Failed to search similar images by embedding: {str(e)}")
        raise


def get_records_by_session(session_id: str) -> Dict[str, Any]:
    """
    Get all records associated with a session ID.
    Returns a dictionary containing prompt, input image, and output image records.
    If input image is not found by session_id, try to fetch by batch_id from prompt/output image metadata (Python-side filtering).
    """
    try:
        if not connect_to_milvus():
            return {
                "prompt_record": None,
                "input_image_record": None,
                "output_image_record": None
            }
        
        # Get collections
        prompt_collection = Collection(PROMPT_COLLECTION)
        input_collection = Collection(INPUT_IMAGE_COLLECTION)
        output_collection = Collection(OUTPUT_IMAGE_COLLECTION)
        
        # Load collections
        prompt_collection.load()
        input_collection.load()
        output_collection.load()
        
        # Query prompt record
        prompt_records = prompt_collection.query(
            expr=f'session_id == "{session_id}"',
            output_fields=["*"]
        )
        # Query output image record
        output_records = output_collection.query(
            expr=f'session_id == "{session_id}"',
            output_fields=["*"]
        )
        # Query input image record by session_id
        input_records = input_collection.query(
            expr=f'session_id == "{session_id}"',
            output_fields=["*"]
        )
        # Process records
        prompt_record = prompt_records[0] if prompt_records else None
        output_record = output_records[0] if output_records else None
        input_record = input_records[0] if input_records else None
        # Parse metadata for each record
        if prompt_record:
            prompt_record["metadata"] = json.loads(prompt_record.get("metadata", "{}"))
        if input_record:
            input_record["metadata"] = json.loads(input_record.get("metadata", "{}"))
        if output_record:
            output_record["metadata"] = json.loads(output_record.get("metadata", "{}"))
        # If input image not found by session_id, try by batch_id (Python-side filtering)
        if not input_record:
            batch_id = None
            if prompt_record and "batch_id" in prompt_record["metadata"]:
                batch_id = prompt_record["metadata"].get("batch_id")
            elif output_record and "batch_id" in output_record["metadata"]:
                batch_id = output_record["metadata"].get("batch_id")
            if batch_id:
                # Fetch a batch of input images and filter in Python
                all_input_records = input_collection.query(
                    expr="status != ''",
                    output_fields=["*"]
                )
                for rec in all_input_records:
                    meta = json.loads(rec.get("metadata", "{}"))
                    if meta.get("batch_id") == batch_id:
                        rec["metadata"] = meta
                        input_record = rec
                        break
        return {
            "prompt_record": prompt_record,
            "input_image_record": input_record,
            "output_image_record": output_record
        }
    except Exception as e:
        print(f"Error fetching session records: {str(e)}")
        return {
            "prompt_record": None,
            "input_image_record": None,
            "output_image_record": None
        }


def get_records_by_model(model_used: str, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieve all records for a specific model."""
    try:
        output_collection = Collection(OUTPUT_IMAGE_COLLECTION)
        
        # Query output image records
        results = output_collection.query(
            expr=f'model_used == "{model_used}"',
            output_fields=[
                "id",
                "session_id",
                "output_image_path",
                "category",
                "timestamp",
                "status",
                "version",
                "metadata"
            ],
            limit=limit
        )
        
        # For each output record, get associated prompt and input image
        complete_records = []
        for result in results:
            session_id = result["session_id"]
            session_records = get_records_by_session(session_id)
            complete_records.append({
                "output_record": result,
                "prompt_record": session_records["prompt_record"],
                "input_image_record": session_records["input_image_record"]
            })
            
        return complete_records
    except Exception as e:
        print(f"❌ Failed to get records by model: {str(e)}")
        raise


def search_by_batch_id(batch_id: str, top_k: int = 10) -> List[Dict]:
    """
    Search for all generations in a batch using the batch_id.
    
    Args:
        batch_id (str): The batch ID to search for
        top_k (int): Number of results to return
        
    Returns:
        List[Dict]: List of matching records with their details
    """
    try:
        if not connect_to_milvus():
            return []
            
        # Search in prompt collection
        prompt_results = search_collection(
            PROMPT_COLLECTION,
            batch_id,
            top_k=top_k,
            search_field="metadata.batch_id"
        )
        
        # Search in image collection
        image_results = search_collection(
            INPUT_IMAGE_COLLECTION,
            batch_id,
            top_k=top_k,
            search_field="metadata.batch_id"
        )
        
        # Combine and sort results by generation_index
        all_results = []
        for result in prompt_results + image_results:
            if "metadata" in result and "generation_index" in result["metadata"]:
                all_results.append(result)
        
        # Sort by generation_index
        all_results.sort(key=lambda x: x["metadata"].get("generation_index", 0))
        
        return all_results
    except Exception as e:
        print(f"⚠️ [WARNING] Batch search failed: {e}")
        return []


def search_by_prompt_similarity(prompt: str, top_k: int = 5) -> List[Dict]:
    """
    Search for similar prompts using text embedding.
    
    Args:
        prompt (str): The prompt to search for
        top_k (int): Number of results to return
        
    Returns:
        List[Dict]: List of similar prompts with their details
    """
    try:
        if not connect_to_milvus():
            return []
            
        # Generate embedding for the search prompt
        embedding = generate_text_embedding(prompt)
        if embedding is None or len(embedding) == 0:
            return []
            
        # Search in prompt collection
        collection = Collection(PROMPT_COLLECTION)
        collection.load()
        
        # Prepare search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Search in Milvus
        results = collection.search(
            data=[embedding.tolist()],
            anns_field="prompt_embedding",
            param=search_params,
            limit=top_k,
            output_fields=["*"]  # Get all fields
        )
        
        # Format results
        similar_prompts = []
        for hits in results:
            for hit in hits:
                # Get the full record
                record = collection.query(
                    expr=f'id == {hit.id}',
                    output_fields=["*"]
                )[0]
                
                # Parse metadata
                metadata = json.loads(record.get("metadata", "{}"))
                
                # Get related records
                session_id = record["session_id"]
                session_records = get_records_by_session(session_id)
                
                # Create result dictionary
                result_dict = {
                    "id": hit.id,
                    "distance": float(hit.distance),
                    "session_id": session_id,
                    "user_prompt": record.get("user_prompt"),
                    "enhanced_prompt": record.get("enhanced_prompt"),
                    "final_prompt": record.get("final_prompt"),
                    "timestamp": record.get("timestamp"),
                    "status": record.get("status"),
                    "version": record.get("version"),
                    "metadata": metadata,
                    "batch_id": metadata.get("batch_id"),
                    "generation_index": metadata.get("generation_index"),
                    "prompt_record": record,
                    "input_image_record": session_records.get("input_image_record"),
                    "output_image_record": session_records.get("output_image_record")
                }
                
                # Add to results
                similar_prompts.append(result_dict)
        
        # Sort by distance (lower is better)
        similar_prompts.sort(key=lambda x: x["distance"])
        
        return similar_prompts
    except Exception as e:
        print(f"⚠️ [WARNING] Prompt search failed: {e}")
        return []


def search_similar_images(image: Image.Image, top_k: int = 5) -> List[Dict]:
    """
    Search for similar images using image embedding.
    
    Args:
        image (Image.Image): The image to search for
        top_k (int): Number of results to return
        
    Returns:
        List[Dict]: List of similar images with their details
    """
    try:
        if not connect_to_milvus():
            return []
            
        # Generate embedding for the search image
        embedding = generate_image_embedding(image)
        if not embedding:
            return []
            
        # Search in image collection
        results = search_collection(
            OUTPUT_IMAGE_COLLECTION,
            embedding,
            top_k=top_k,
            search_field="output_image_embedding"
        )
        
        # Add batch information to results
        for result in results:
            if "metadata" in result and "batch_id" in result["metadata"]:
                batch_id = result["metadata"]["batch_id"]
                # Get related generations from the same batch
                batch_results = search_by_batch_id(batch_id)
                result["related_generations"] = batch_results
        
        return results
    except Exception as e:
        print(f"⚠️ [WARNING] Image search failed: {e}")
        return []


def update_record_status(
    collection_name: str,
    session_id: str,
    status: str,
    metadata: Optional[dict] = None
) -> bool:
    """Update the status and metadata of a record."""
    try:
        collection = Collection(collection_name)
        
        # First get the record using session_id
        results = collection.query(
            expr=f'session_id == "{session_id}"',
            output_fields=["*"]
        )
        
        if not results:
            print(f"⚠️ Warning: No record found with session_id {session_id}")
            return False
            
        # Get the existing record data
        existing_data = results[0]
        
        # Update only the status and metadata fields
        existing_data["status"] = status
        if metadata:
            existing_data["metadata"] = json.dumps(metadata)
            
        # Delete the existing record
        collection.delete(f'session_id == "{session_id}"')
        
        # Insert the updated record
        collection.insert([existing_data])
        collection.flush()
        
        # Log the operation
        log_operation("UPDATE_STATUS", {
            "collection": collection_name,
            "session_id": session_id,
            "record_id": existing_data.get("id"),
            "status": status,
            "timestamp": int(time.time())
        })
        
        return True
    except Exception as e:
        print(f"❌ Failed to update record status: {str(e)}")
        raise


def _check_for_duplicate(
    input_prompt: str,
    output_image_path: str,
    threshold: float = 0.95,
    check_input_images: bool = False
) -> Optional[int]:
    """
    Check if a similar generation already exists.
    
    Args:
        input_prompt: The input prompt
        output_image_path: Path to the output image
        threshold: Similarity threshold
        check_input_images: If True, check against input images, otherwise check against output images
    """
    try:
        # Generate embedding for output image
        image_embedding = generate_image_embedding(output_image_path)

        # Determine which field to search
        anns_field = "input_image_embedding" if check_input_images else "output_image_embedding"

        # Search for similar images
        collection = Collection(INPUT_IMAGE_COLLECTION)
        collection.load()
        results = collection.search(
            data=[image_embedding.tolist()],
            anns_field=anns_field,
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


def get_all_session_ids() -> List[str]:
    """Get all unique session IDs from all collections."""
    try:
        # Ensure collections are loaded
        if not ensure_collections_loaded():
            raise Exception("Failed to ensure collections are loaded")

        # Get session IDs from each collection
        session_ids = set()
        
        for collection_name in [PROMPT_COLLECTION, INPUT_IMAGE_COLLECTION, OUTPUT_IMAGE_COLLECTION]:
            collection = Collection(collection_name)
            results = collection.query(
                expr="session_id != ''",
                output_fields=["session_id"]
            )
            session_ids.update(result["session_id"] for result in results)

        return sorted(list(session_ids))
    except Exception as e:
        print(f"❌ Failed to get session IDs: {str(e)}")
        raise


def search_collection(collection_name: str, query: Union[str, np.ndarray], top_k: int = 5, search_field: str = None) -> List[Dict]:
    """
    Generic search function for Milvus collections.
    
    Args:
        collection_name (str): Name of the collection to search
        query (Union[str, np.ndarray]): Query string or embedding
        top_k (int): Number of results to return
        search_field (str): Field to search on (for metadata fields, use dot notation)
        
    Returns:
        List[Dict]: List of matching records
    """
    try:
        collection = Collection(collection_name)
        collection.load()
        
        # Prepare search parameters
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        # Handle different types of queries
        if isinstance(query, str):
            if search_field and "." in search_field:
                # Search in metadata field
                expr = f'json_contains(metadata, \'{{"{search_field}": "{query}"}}\')'
                results = collection.query(
                    expr=expr,
                    output_fields=["*"],
                    limit=top_k
                )
                return results
            else:
                # Text search
                embedding = generate_text_embedding(query)
                if embedding is None:
                    return []
                search_data = [embedding.tolist()]
        else:
            # Vector search
            search_data = [query.tolist()]
        
        # Perform vector search
        results = collection.search(
            data=search_data,
            anns_field=search_field or "embedding",
            param=search_params,
            limit=top_k,
            output_fields=["*"]
        )
        
        # Format results
        formatted_results = []
        for hits in results:
            for hit in hits:
                record = collection.query(
                    expr=f'id == {hit.id}',
                    output_fields=["*"]
                )[0]
                
                # Parse metadata
                metadata = json.loads(record.get("metadata", "{}"))
                record["metadata"] = metadata
                
                formatted_results.append({
                    "id": hit.id,
                    "distance": float(hit.distance),
                    **record
                })
        
        return formatted_results
    except Exception as e:
        print(f"⚠️ [WARNING] Collection search failed: {e}")
        return []


# Initialize Milvus connection and collections on import
try:
    connect_to_milvus()
    ensure_collections()  # This will only create collections if they don't exist
except Exception as e:
    print(f"⚠️ Failed to initialize Milvus: {str(e)}")
