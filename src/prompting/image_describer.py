"""
Image description and analysis module for the prompt enhancement system.
Provides secure and configurable image analysis using vision-language models.
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Any, Union

import torch
import openai
from PIL import Image

from .model_manager import get_model_manager
from .constants import (
    PromptConfig,
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_SYSTEM_CONFIG,
)

# === Logging Setup ===
logger = logging.getLogger(__name__)

# === Constants ===
VISION_DEVICE = os.getenv("VISION_DEVICE", "cpu")
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", 120))
DEFAULT_VISION_PROMPT = "a photograph of"
DEFAULT_MODEL_NAME = "gpt-4"

# === OpenAI Setup ===
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def clean_caption(text: str) -> str:
    """
    Clean and format a caption text.

    Args:
        text (str): Raw caption text

    Returns:
        str: Cleaned caption text
    """
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(' "\n').capitalize()

def analyze_caption_with_gpt(
    caption: str,
    config: Optional[PromptConfig] = None,
    custom_role: Optional[str] = None
) -> Dict[str, str]:
    """
    Analyze a caption using GPT to extract structured information.

    Args:
        caption (str): Image caption to analyze
        config (Optional[PromptConfig]): Configuration for analysis
        custom_role (Optional[str]): Custom system role override

    Returns:
        Dict[str, str]: Structured information about the image
    """
    logger.debug(f"Analyzing caption: {caption[:100]}...")

    # Get system role from config or use default
    system_role = (
        custom_role or
        (config and config.get("image_analyzer_role")) or
        DEFAULT_SYSTEM_CONFIG["roles"]["image_describer"]
    )

    system_message = (
        f"{system_role}\n"
        "Given a caption, return three things:\n"
        "1. The main product or object described\n"
        "2. The visual style or medium\n"
        "3. The inferred intent\n"
        "Only return a JSON dictionary with keys: product, style, intent."
    )

    user_message = f"Image caption: {caption}"
    try:
        response = client.chat.completions.create(
            model=config.model_name if config else DEFAULT_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()
        
        # Safely parse JSON response
        try:
            result = json.loads(content)
            if not isinstance(result, dict):
                logger.warning("GPT returned non-dictionary JSON")
                return {}
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse GPT response as JSON: {e}")
            return {}

        logger.debug("GPT analysis results:")
        for key, value in result.items():
            logger.debug(f"  - {key}: {value}")

        return result
    except Exception as e:
        logger.error(f"Failed to analyze caption: {e}")
        return {}

class VisionDescriber:
    """
    Describes images using a vision-language model.
    Supports multiple models and configurable parameters.
    """

    def __init__(
        self,
        device: str = VISION_DEVICE,
        max_tokens: int = VISION_MAX_TOKENS,
        model_name: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        config: Optional[PromptConfig] = None
    ):
        """
        Initialize the vision describer.

        Args:
            device (str): Device to run the model on
            max_tokens (int): Maximum tokens for generation
            model_name (Optional[str]): Name of the model to use
            model_config (Optional[Dict[str, Any]]): Additional model configuration
            config (Optional[PromptConfig]): Global configuration
        """
        self.device = device
        self.max_tokens = max_tokens
        self.model_name = model_name
        self.model_config = model_config or {}
        self.config = config
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the vision model and processor."""
        if self.processor is None or self.model is None:
            logger.info(f"Loading vision model on {self.device}")
            try:
                model_manager = get_model_manager()
                self.model, self.processor = model_manager.load_model(
                    device=self.device
                )
                logger.info("Vision model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load vision model: {e}")
                raise

    def describe_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None
    ) -> str:
        """
        Describe an image using the vision model.

        Args:
            image (Image.Image): Image to describe
            prompt (Optional[str]): Custom prompt for the model

        Returns:
            str: Generated description
        """
        try:
            # Use prompt from config or default
            prompt = (
                prompt or
                (self.config and self.config.get("vision_prompt")) or
                DEFAULT_VISION_PROMPT
            )
            logger.debug(f"Describing image with prompt: {prompt[:100]}...")

            # Process the image
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
                self.device
            )

            # Generate caption
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    num_beams=5,
                    length_penalty=1.0,
                )

            # Decode the generated caption
            caption = self.processor.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True
            )
            cleaned = clean_caption(caption)

            logger.debug(f"Generated caption: {cleaned}")
            return cleaned
        except Exception as e:
            logger.error(f"Failed to describe image: {e}")
            return ""

def describe_uploaded_images(
    images: List[Image.Image],
    prompt_override: str = "",
    describer: Optional[VisionDescriber] = None,
    config: Optional[PromptConfig] = None
) -> Dict[str, str]:
    """
    Describe multiple uploaded images and analyze their content.

    Args:
        images (List[Image.Image]): List of images to describe
        prompt_override (str): Optional custom prompt
        describer (Optional[VisionDescriber]): Optional vision describer instance
        config (Optional[PromptConfig]): Configuration for analysis

    Returns:
        Dict[str, str]: Structured information about the images
    """
    if not images:
        logger.warning("No images provided")
        return {
            "style_description": "",
            "full_caption": "",
            "product_description": "",
            "intent_description": "",
        }

    logger.info(f"Processing {len(images)} images")
    
    # Initialize describer if not provided
    if describer is None:
        describer = VisionDescriber(
            config=config or DEFAULT_PROMPT_CONFIG
        )

    # Get prompt from config or use override
    prompt = (
        prompt_override or
        (config and config.get("vision_prompt")) or
        DEFAULT_VISION_PROMPT
    )
    
    captions = []

    for i, img in enumerate(images, 1):
        logger.debug(f"Processing image {i}/{len(images)}")
        caption = describer.describe_image(img, prompt)
        if caption:
            captions.append(caption)

    combined_caption = "; ".join(sorted(set(captions)))
    logger.debug(f"Combined caption: {combined_caption}")

    gpt_insights = analyze_caption_with_gpt(
        combined_caption,
        config=config,
        custom_role=config.get("image_analyzer_role") if config else None
    )

    result = {
        "full_caption": combined_caption,
        "style_description": gpt_insights.get("style", ""),
        "product_description": gpt_insights.get("product", ""),
        "intent_description": gpt_insights.get("intent", ""),
    }

    logger.debug("Final analysis results:")
    for key, value in result.items():
        logger.debug(f"  - {key}: {value}")

    return result
