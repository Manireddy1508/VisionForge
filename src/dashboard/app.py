import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import json
from datetime import datetime, timedelta
import os
from pathlib import Path
import time
from typing import List, Dict, Any, Union
import logging

# Import Milvus utilities
import sys
sys.path.append(str(Path(__file__).parent.parent))
from milvus_utils import (
    connect_to_milvus,
    get_records_by_session,
    search_similar_prompts,
    search_similar_images,
    search_similar_images_by_embedding,
    search_by_prompt_similarity,
    get_all_session_ids,
    PROMPT_COLLECTION,
    INPUT_IMAGE_COLLECTION,
    OUTPUT_IMAGE_COLLECTION,
    generate_image_embedding,
    Collection,
    connections
)
from src.dashboard.utils import (
    format_timestamp,
    format_embedding_stats,
    get_all_full_session_records
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Image Generation Dashboard",
    page_icon="🎨",
    layout="wide"
)

# Initialize session state
if 'selected_session' not in st.session_state:
    st.session_state.selected_session = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'search_log' not in st.session_state:
    st.session_state.search_log = []

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Calculate cosine similarity between two embeddings."""
    if not embedding1 or not embedding2:
        return 0.0
    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    except Exception as e:
        logger.error(f"Error calculating similarity: {str(e)}")
        return 0.0

def display_image(image_path: str, caption: str):
    """Display an image with a caption."""
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path)
            st.image(image, caption=caption, use_column_width=True)
        else:
            st.warning(f"Image not found: {image_path}")
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")

def display_metadata(metadata: Union[str, dict]):
    """Display metadata in a formatted way."""
    try:
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        for key, value in metadata.items():
            st.write(f"{key}: {value}")
    except:
        st.write("No metadata available")

def get_metrics():
    """Get dashboard metrics from Milvus collections."""
    try:
        # Get total sessions
        session_ids = get_all_session_ids()
        total_sessions = len(session_ids)
        
        # Get counts from collections
        prompt_collection = Collection(PROMPT_COLLECTION)
        output_collection = Collection(OUTPUT_IMAGE_COLLECTION)
        
        # Get status counts
        status_counts = {
            "completed": 0,
            "failed": 0,
            "pending": 0
        }
        
        results = output_collection.query(
            expr="status != ''",
            output_fields=["status"]
        )
        for result in results:
            status = result["status"].lower()
            if status in status_counts:
                status_counts[status] += 1
        
        return {
            "total_sessions": total_sessions,
            "total_prompts": prompt_collection.num_entities,
            "completed_generations": status_counts["completed"],
            "failed_generations": status_counts["failed"]
        }
    except Exception as e:
        st.error(f"Error getting metrics: {str(e)}")
        return {
            "total_sessions": 0,
            "total_prompts": 0,
            "completed_generations": 0,
            "failed_generations": 0
        }

def get_distinct_values(collection_name, field_name):
    """Get distinct values for a field in a collection."""
    try:
        collection = Collection(collection_name)
        results = collection.query(
            expr=f"{field_name} != ''",
            output_fields=[field_name]
        )
        return sorted(list(set(result[field_name] for result in results)))
    except Exception as e:
        st.error(f"Error getting distinct values: {str(e)}")
        return []

def get_all_full_session_records(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch and join all session records from Milvus collections.
    Returns a list of flattened dictionaries with all relevant fields.
    """
    try:
        # Connect to Milvus
        connections.connect(host='standalone', port=19530)
        
        # Get collections
        prompt_collection = Collection(PROMPT_COLLECTION)
        input_collection = Collection(INPUT_IMAGE_COLLECTION)
        output_collection = Collection(OUTPUT_IMAGE_COLLECTION)
        
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
            metadata = json.loads(record.get('metadata', '{}'))
            batch_id = metadata.get('batch_id', 'N/A')
            generation_index = metadata.get('generation_index', 'N/A')
            session_records[session_id] = {
                "session_id": session_id,
                "batch_id": batch_id,
                "generation_index": generation_index,
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
                metadata = json.loads(record.get('metadata', '{}'))
                session_records[session_id].update({
                    "input_image_path": record.get('input_image_path', 'N/A'),
                    "input_timestamp": record.get('timestamp', 0),
                    "input_status": record.get('status', 'N/A'),
                    "input_embedding": record.get('input_image_embedding', []),
                    "input_metadata": metadata
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
            metadata = json.loads(record.get('metadata', '{}'))
            if session_id in session_records:
                # Fallback: if batch_id or generation_index is missing, get from output image metadata
                if session_records[session_id]["batch_id"] == 'N/A' and metadata.get('batch_id'):
                    session_records[session_id]["batch_id"] = metadata.get('batch_id')
                if session_records[session_id]["generation_index"] == 'N/A' and metadata.get('generation_index'):
                    session_records[session_id]["generation_index"] = metadata.get('generation_index')
                session_records[session_id].update({
                    "output_image_path": record.get('output_image_path', 'N/A'),
                    "model_used": record.get('model_used', 'N/A'),
                    "category": record.get('category', 'N/A'),
                    "output_timestamp": record.get('timestamp', 0),
                    "output_status": record.get('status', 'N/A'),
                    "output_embedding": record.get('output_image_embedding', []),
                    "output_metadata": metadata
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

def main():
    st.title("🎨 Image Generation Dashboard")
    
    # Initialize session state for auto-refresh
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
    # Auto-refresh every 30 seconds
    current_time = time.time()
    if current_time - st.session_state.last_refresh > 30:
        st.session_state.last_refresh = current_time
        st.experimental_rerun()
    
    # Connect to Milvus
    try:
        connect_to_milvus()
    except Exception as e:
        st.error(f"Failed to connect to Milvus: {str(e)}")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Session Details", "🖼️ Image Search", "🔎 Prompt Search", "📊 Data Table"])
    
    with tab1:
        # Manual refresh button
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🔄 Refresh"):
                st.session_state.last_refresh = time.time()
                st.experimental_rerun()
        
        # Session selector
        session_id = st.text_input("Enter Session ID", key="session_id")
        
        if session_id:
            try:
                # Get records for the session
                records = get_records_by_session(session_id)
                
                # Create three columns for the layout
                col1, col2, col3 = st.columns(3)
                
                # Display prompt information
                with col1:
                    st.subheader("Prompt Information")
                    if records["prompt_record"]:
                        prompt = records["prompt_record"]
                        metadata = prompt.get("metadata", {})
                        # Fallback to output image metadata if missing
                        output_img = records.get("output_image_record")
                        output_metadata = output_img.get("metadata", {}) if output_img else {}
                        batch_id = metadata.get("batch_id") or output_metadata.get("batch_id")
                        generation_index = metadata.get("generation_index") or output_metadata.get("generation_index")
                        st.text_area("User Prompt", prompt.get("user_prompt", ""), height=100)
                        st.text_area("Enhanced Prompt", prompt.get("enhanced_prompt", ""), height=100)
                        st.text_area("Final Prompt", prompt.get("final_prompt", ""), height=100)
                        if batch_id:
                            st.write("Batch ID:", batch_id)
                        if generation_index:
                            st.write("Generation Index:", generation_index)
                        st.write("Timestamp:", format_timestamp(prompt["timestamp"]))
                        st.write("Status:", prompt["status"])
                        st.write("Version:", prompt["version"])
                        st.write("Embedding Statistics:")
                        st.json(format_embedding_stats(prompt.get("prompt_embedding", [])))
                        st.write("Metadata:")
                        display_metadata(metadata)
                        st.caption(f"Prompt Session ID: {prompt.get('session_id', 'N/A')}")
                        st.caption(f"Prompt Batch ID: {batch_id if batch_id else 'N/A'}")
                    else:
                        st.info("No prompt record found for this session")
                
                # Display input image information
                with col2:
                    st.subheader("Input Image")
                    if records["input_image_record"]:
                        input_img = records["input_image_record"]
                        metadata = input_img.get("metadata", {})
                        display_image(input_img["input_image_path"], "Input Image")
                        if "batch_id" in metadata:
                            st.write("Batch ID:", metadata["batch_id"])
                        st.write("Timestamp:", format_timestamp(input_img["timestamp"]))
                        st.write("Status:", input_img["status"])
                        st.write("Version:", input_img["version"])
                        st.write("Embedding Statistics:")
                        st.json(format_embedding_stats(input_img.get("input_image_embedding", [])))
                        st.write("Metadata:")
                        display_metadata(metadata)
                        st.caption(f"Input Image Session ID: {input_img.get('session_id', 'N/A')}")
                        st.caption(f"Input Image Batch ID: {metadata.get('batch_id', 'N/A')}")
                    else:
                        st.info("No input image record found for this session")
                
                # Display output image information
                with col3:
                    st.subheader("Output Image")
                    if records["output_image_record"]:
                        output_img = records["output_image_record"]
                        metadata = output_img.get("metadata", {})
                        display_image(output_img["output_image_path"], "Output Image")
                        st.write("Batch ID:", metadata.get("batch_id", "N/A"))
                        st.write("Generation Index:", metadata.get("generation_index", "N/A"))
                        st.write("Model Used:", output_img["model_used"])
                        st.write("Category:", output_img["category"])
                        st.write("Timestamp:", format_timestamp(output_img["timestamp"]))
                        st.write("Status:", output_img["status"])
                        st.write("Version:", output_img["version"])
                        st.write("Embedding Statistics:")
                        st.json(format_embedding_stats(output_img.get("output_image_embedding", [])))
                        st.write("Metadata:")
                        display_metadata(metadata)
                        st.caption(f"Output Image Session ID: {output_img.get('session_id', 'N/A')}")
                        st.caption(f"Output Image Batch ID: {metadata.get('batch_id', 'N/A')}")
                    else:
                        st.info("No output image record found for this session")
                
                # Display search results if available
                if hasattr(st.session_state, 'search_results'):
                    st.subheader("Search Results")
                    for result in st.session_state.search_results:
                        with st.expander(f"Result (Distance: {result['distance']:.4f})"):
                            # Show similarity score with color
                            similarity = 1 - result['distance']
                            similarity_clamped = max(0.0, min(1.0, similarity))
                            st.progress(similarity_clamped)
                            st.write(f"Similarity: {similarity_clamped:.2%}")
                            
                            if "user_prompt" in result:
                                st.write("Prompt:", result["user_prompt"])
                            if "input_image_path" in result:
                                display_image(result["input_image_path"], "Input Image")
                            if "output_image_path" in result:
                                display_image(result["output_image_path"], "Output Image")
            except Exception as e:
                st.error(f"Error loading session data: {str(e)}")
    
    with tab2:
        st.subheader("Search Similar Images")
        
        # File uploader for search image
        search_image = st.file_uploader("Upload an image to search", type=["png", "jpg", "jpeg"])
        
        if search_image:
            try:
                # Convert uploaded file to PIL Image
                image = Image.open(search_image)
                st.image(image, caption="Search Image", use_column_width=True)
                
                # Generate embedding and search
                embedding = generate_image_embedding(image)
                if embedding is not None:
                    results = search_similar_images_by_embedding(embedding, top_k=5)
                    
                    if results:
                        st.subheader("Similar Images")
                        for result in results:
                            with st.expander(f"Result (Distance: {result['distance']:.4f})"):
                                # Show similarity score with color
                                similarity = 1 - result['distance']
                                similarity_clamped = max(0.0, min(1.0, similarity))
                                st.progress(similarity_clamped)
                                st.write(f"Similarity: {similarity_clamped:.2%}")
                                
                                if "input_image_path" in result and result["input_image_path"]:
                                    display_image(result["input_image_path"], "Input Image")
                                if "output_image_path" in result and result["output_image_path"]:
                                    display_image(result["output_image_path"], "Output Image")
                    else:
                        st.info("No similar images found")
            except Exception as e:
                st.error(f"Error searching images: {str(e)}")

    with tab3:
        st.subheader("Search Similar Prompts")
        prompt_query = st.text_input("Enter a prompt to search for similar prompts:", key="prompt_search_input")
        top_k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        if st.button("Search Prompts") and prompt_query:
            try:
                results = search_by_prompt_similarity(prompt_query, top_k=top_k)
                if results:
                    st.subheader("Similar Prompts")
                    for result in results:
                        with st.expander(f"Prompt: {result.get('user_prompt', '')[:60]}... (Similarity: {1 - result.get('distance', 0):.2%})"):
                            # Display prompt information
                            st.write(f"**User Prompt:** {result.get('user_prompt', '')}")
                            st.write(f"**Enhanced Prompt:** {result.get('enhanced_prompt', '')}")
                            st.write(f"**Final Prompt:** {result.get('final_prompt', '')}")
                            st.write(f"**Timestamp:** {format_timestamp(result.get('timestamp', 0))}")
                            st.write(f"**Status:** {result.get('status', '')}")
                            st.write(f"**Version:** {result.get('version', '')}")
                            
                            # Display metadata
                            metadata = result.get('metadata', {})
                            st.write("**Batch ID:**", metadata.get('batch_id', 'N/A'))
                            st.write("**Generation Index:**", metadata.get('generation_index', 'N/A'))
                            
                            # Display input image if available
                            input_record = result.get('input_image_record')
                            if input_record and input_record.get('input_image_path'):
                                st.write("**Input Image:**")
                                display_image(input_record['input_image_path'], "Input Image")
                                input_metadata = input_record.get('metadata', {})
                                st.write("**Input Batch ID:**", input_metadata.get('batch_id', 'N/A'))
                                st.write("**Input Generation Index:**", input_metadata.get('generation_index', 'N/A'))
                                st.write("**Input Status:**", input_record.get('status', 'N/A'))
                                st.write("**Input Version:**", input_record.get('version', 'N/A'))
                                st.write("**Input Timestamp:**", format_timestamp(input_record.get('timestamp', 0)))
                            
                            # Display output image if available
                            output_record = result.get('output_image_record')
                            if output_record and output_record.get('output_image_path'):
                                st.write("**Output Image:**")
                                display_image(output_record['output_image_path'], "Output Image")
                                output_metadata = output_record.get('metadata', {})
                                st.write("**Output Batch ID:**", output_metadata.get('batch_id', 'N/A'))
                                st.write("**Output Generation Index:**", output_metadata.get('generation_index', 'N/A'))
                                st.write("**Model Used:**", output_record.get('model_used', 'N/A'))
                                st.write("**Category:**", output_record.get('category', 'N/A'))
                                st.write("**Output Status:**", output_record.get('status', 'N/A'))
                                st.write("**Output Version:**", output_record.get('version', 'N/A'))
                                st.write("**Output Timestamp:**", format_timestamp(output_record.get('timestamp', 0)))
                else:
                    st.info("No similar prompts found.")
            except Exception as e:
                st.error(f"Error searching prompts: {str(e)}")
                logger.error(f"Prompt search error: {str(e)}", exc_info=True)

    with tab4:
        st.subheader("All Session Records")
        
        try:
            # Get all session records
            records = get_all_full_session_records(limit=100)
            
            if not records:
                st.info("No records found in the database.")
                return
            
            # Convert to DataFrame
            df = pd.DataFrame(records)
            
            # Format timestamps
            df['prompt_timestamp'] = df['prompt_timestamp'].apply(format_timestamp)
            df['input_timestamp'] = df['input_timestamp'].apply(format_timestamp)
            df['output_timestamp'] = df['output_timestamp'].apply(format_timestamp)
            
            # Format similarity scores as percentages
            df['prompt_input_similarity'] = df['prompt_input_similarity'].apply(lambda x: f"{x:.2%}")
            df['prompt_output_similarity'] = df['prompt_output_similarity'].apply(lambda x: f"{x:.2%}")
            
            # Format embedding statistics
            df['input_stats'] = df['input_embedding_stats'].apply(
                lambda x: f"μ={x['mean']:.4f} σ={x['std']:.4f} min={x['min']:.4f} max={x['max']:.4f}"
            )
            df['output_stats'] = df['output_embedding_stats'].apply(
                lambda x: f"μ={x['mean']:.4f} σ={x['std']:.4f} min={x['min']:.4f} max={x['max']:.4f}"
            )
            
            # Clean up user prompt display
            def clean_prompt(x):
                if isinstance(x, str):
                    if '<gradio.components.textbox.Textbox' in x:
                        # Extract the value between the last '>' and the end
                        parts = x.split('>')
                        if len(parts) > 1:
                            return parts[-1].strip()
                return x
            
            df['user_prompt'] = df['user_prompt'].apply(clean_prompt)
            
            # Select and rename columns for display
            display_df = df[[
                'session_id',
                'batch_id',
                'generation_index',
                'user_prompt',
                'final_prompt',
                'input_image_path',
                'output_image_path',
                'model_used',
                'category',
                'prompt_status',
                'input_status',
                'output_status',
                'prompt_timestamp',
                'input_timestamp',
                'output_timestamp',
                'prompt_input_similarity',
                'prompt_output_similarity',
                'input_stats',
                'output_stats'
            ]].rename(columns={
                'prompt_status': 'Prompt Status',
                'input_status': 'Input Status',
                'output_status': 'Output Status',
                'prompt_timestamp': 'Prompt Time',
                'input_timestamp': 'Input Time',
                'output_timestamp': 'Output Time',
                'prompt_input_similarity': 'Prompt→Input Sim',
                'prompt_output_similarity': 'Prompt→Output Sim',
                'input_stats': 'Input Stats',
                'output_stats': 'Output Stats'
            })
            
            # Display the dataframe
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Show record count
            st.write(f"Showing {len(display_df)} records")
            
        except Exception as e:
            st.error(f"Error loading data table: {str(e)}")

if __name__ == "__main__":
    main() 