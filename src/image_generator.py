import os
import logging
from typing import Dict, Any, List, Optional
import openai
from PIL import Image
import io
import base64
from datetime import datetime
import json
from google.cloud import storage
from src.prompting.constants import (
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageGenerator:
    def __init__(self):
        """Initialize the image generator with OpenAI API key."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.model = os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # Initialize Google Cloud Storage client
        self.gcs_client = storage.Client()
        self.bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not self.bucket_name:
            raise ValueError("GCS_BUCKET_NAME environment variable is not set")
        
        self.bucket = self.gcs_client.bucket(self.bucket_name)
    
    def generate_image(
        self,
        prompt: str,
        guidance_scale: float = 7.5,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 50,
        seed: int = -1,
        scheduler: str = "DPMSolverMultistep",
    ) -> Image.Image:
        """
        Generate an image using OpenAI's image generation API.
        
        Args:
            prompt: The text prompt describing the image to generate
            guidance_scale: How closely to follow the prompt (higher = more precise)
            width: Image width in pixels
            height: Image height in pixels
            num_inference_steps: Number of denoising steps
            seed: Random seed (-1 for random)
            scheduler: The scheduler to use for generation
            
        Returns:
            PIL Image object
        """
        try:
            # Format prompt
            final_prompt = f"{PROMPT_PREFIX}{prompt}{PROMPT_SUFFIX}"
            
            # Generate image
            response = self.client.images.generate(
                model=self.model,
                prompt=final_prompt,
                size=f"{width}x{height}",
                quality="standard",
                n=1,
                response_format="b64_json",
            )
            
            # Convert base64 to image
            image_data = base64.b64decode(response.data[0].b64_json)
            image = Image.open(io.BytesIO(image_data))
            
            # Save to GCS
            self._save_to_gcs(image, prompt)
            
            return image
            
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            raise
    
    def _save_to_gcs(self, image: Image.Image, prompt: str) -> None:
        """
        Save the generated image to Google Cloud Storage.
        
        Args:
            image: PIL Image object to save
            prompt: The prompt used to generate the image
        """
        try:
            # Create a unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_images/{timestamp}.png"
            
            # Save image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_byte_arr = img_byte_arr.getvalue()
            
            # Upload to GCS
            blob = self.bucket.blob(filename)
            blob.upload_from_string(
                img_byte_arr,
                content_type="image/png",
                metadata={"prompt": prompt},
            )
            
            logger.info(f"Image saved to GCS: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving image to GCS: {str(e)}")
            raise
