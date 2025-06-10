import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pymilvus import connections, Collection
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_timestamp(timestamp: int) -> str:
    """Format Unix timestamp to readable datetime string."""
    try:
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return "N/A"

def format_embedding_stats(embedding: Optional[List[float]]) -> Dict[str, float]:
    """Calculate statistics for an embedding vector."""
    if not embedding:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    try:
        arr = np.array(embedding)
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }
    except:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}

def calculate_similarity(embedding1: Optional[List[float]], embedding2: Optional[List[float]]) -> float:
    """Calculate cosine similarity between two embeddings."""
    if not embedding1 or not embedding2:
        return 0.0
    try:
        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)
        return float(np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))
    except:
        return 0.0

def get_all_full_session_records(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch and join all session records from Milvus collections.
    Returns a list of flattened dictionaries with all relevant fields.
    """
    try:
        # Connect to Milvus
        connections.connect(host='standalone', port=19530)
        
        # Get collections
        prompt_collection = Collection('prompt_embeddings')
        input_collection = Collection('input_image_embeddings')
        output_collection = Collection('output_image_embeddings')
        
        # Load collections
        prompt_collection.load()
        input_collection.load()
        output_collection.load()
        
        # Query all records
        prompt_records = prompt_collection.query(
            expr="",
            output_fields=[
                "session_id",
                "user_prompt",
                "final_prompt",
                "prompt_embedding",
                "timestamp",
                "status",
                "version",
                "metadata"
            ],
            limit=limit
        )
        
        # Create a dictionary to store joined records
        session_records = {}
        
        # Process prompt records
        for record in prompt_records:
            session_id = record['session_id']
            session_records[session_id] = {
                "session_id": session_id,
                "user_prompt": record.get('user_prompt', 'N/A'),
                "final_prompt": record.get('final_prompt', 'N/A'),
                "prompt_timestamp": record.get('timestamp', 0),
                "prompt_status": record.get('status', 'N/A'),
                "prompt_version": record.get('version', 'N/A'),
                "prompt_embedding": record.get('prompt_embedding', []),
                "input_image_path": 'N/A',
                "output_image_path": 'N/A',
                "model_used": 'N/A',
                "category": 'N/A',
                "input_timestamp": 0,
                "output_timestamp": 0,
                "input_status": 'N/A',
                "output_status": 'N/A',
                "input_embedding": [],
                "output_embedding": []
            }
        
        # Query and join input image records
        input_records = input_collection.query(
            expr="",
            output_fields=[
                "session_id",
                "input_image_path",
                "input_image_embedding",
                "timestamp",
                "status",
                "version",
                "metadata"
            ],
            limit=limit
        )
        
        for record in input_records:
            session_id = record['session_id']
            if session_id in session_records:
                session_records[session_id].update({
                    "input_image_path": record.get('input_image_path', 'N/A'),
                    "input_timestamp": record.get('timestamp', 0),
                    "input_status": record.get('status', 'N/A'),
                    "input_embedding": record.get('input_image_embedding', [])
                })
        
        # Query and join output image records
        output_records = output_collection.query(
            expr="",
            output_fields=[
                "session_id",
                "output_image_path",
                "output_image_embedding",
                "model_used",
                "category",
                "timestamp",
                "status",
                "version",
                "metadata"
            ],
            limit=limit
        )
        
        for record in output_records:
            session_id = record['session_id']
            if session_id in session_records:
                session_records[session_id].update({
                    "output_image_path": record.get('output_image_path', 'N/A'),
                    "model_used": record.get('model_used', 'N/A'),
                    "category": record.get('category', 'N/A'),
                    "output_timestamp": record.get('timestamp', 0),
                    "output_status": record.get('status', 'N/A'),
                    "output_embedding": record.get('output_image_embedding', [])
                })
        
        # Convert to list and calculate additional fields
        records_list = []
        for record in session_records.values():
            # Calculate similarity scores
            prompt_input_sim = calculate_similarity(
                record.get('prompt_embedding', []),
                record.get('input_embedding', [])
            )
            prompt_output_sim = calculate_similarity(
                record.get('prompt_embedding', []),
                record.get('output_embedding', [])
            )
            
            # Calculate embedding statistics
            input_stats = format_embedding_stats(record.get('input_embedding', []))
            output_stats = format_embedding_stats(record.get('output_embedding', []))
            
            # Add calculated fields
            record.update({
                "prompt_input_similarity": prompt_input_sim,
                "prompt_output_similarity": prompt_output_sim,
                "input_embedding_stats": input_stats,
                "output_embedding_stats": output_stats
            })
            
            records_list.append(record)
        
        return records_list
        
    except Exception as e:
        logger.error(f"Error fetching session records: {str(e)}")
        return [] 