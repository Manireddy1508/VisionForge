"""
Core module for enhancing prompts using LLMs and image analysis.
Provides dynamic prompt enhancement with structured templates and configuration.
"""

import os
import ast
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import openai
from PIL import Image
from openai import OpenAI

from prompting.constants import (
    DEFAULT_NUM_PROMPTS,
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_SYSTEM_CONFIG,
    TEMPLATE_FIELDS,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    REQUIRED_ELEMENTS,
    OPTIONAL_ELEMENTS,
)
from prompting.prompt_router import classify_prompt_intent
from prompting.image_describer import describe_uploaded_images
from prompting.prompt_utils import extract_valid_prompts

# === Logging Setup ===
logger = logging.getLogger(__name__)

# === OpenAI Client ===
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@dataclass
class EnhancementContext:
    """Context for prompt enhancement including all relevant information."""
    user_prompt: str
    num_prompts: int = DEFAULT_NUM_PROMPTS
    reference_images: Optional[List[Image.Image]] = None
    style_hint: str = ""
    product_description: str = ""
    image_description: Optional[str] = None
    intent: Optional[str] = None
    config: Any = field(default_factory=lambda: DEFAULT_PROMPT_CONFIG)

def build_dynamic_system_message(context: EnhancementContext) -> str:
    """
    Builds a dynamic system message based on template fields and context.
    
    Args:
        context: EnhancementContext containing all relevant information
        
    Returns:
        str: Complete system message for GPT
    """
    # Build field descriptions
    field_descriptions = []
    for field in TEMPLATE_FIELDS.values():
        field_desc = f"- {field.name}: {field.description}"
        if field.constraints:
            field_desc += f"\n  Constraints: {', '.join(field.constraints)}"
        field_descriptions.append(field_desc)
    
    # Build template structure
    template_structure = " ".join(
        f"[{field.name}]" for field in TEMPLATE_FIELDS.values()
    )
    
    # Build system message
    system_message = f"""
You are a prompt enhancement assistant for AI image generation.

Your task is to transform the user's input into {context.num_prompts} fully structured, highly descriptive prompts suitable for AI image generation.

Each prompt must follow this structure:
{template_structure}

Required elements:
{chr(10).join(field_descriptions)}

Prompt constraints:
- Must be under {context.config.max_length} words
- Must include all required elements
- Optional elements should be included when they enhance the result
- Must be a single sentence (no bullet points, no numbering)
- Must maintain realistic physical proportions
- Must follow the template structure exactly
- Must start with: "{PROMPT_PREFIX}"
- Must end with: "{PROMPT_SUFFIX}"
"""

    # Add context-specific instructions
    if context.style_hint:
        system_message += f"\nVisual style reference: {context.style_hint}"
    if context.product_description:
        system_message += f"\nProduct context: {context.product_description}"
    if context.image_description:
        system_message += f"\nAdditional image context: {context.image_description}"
    
    return system_message.strip()

def enhance_prompt_with_chatgpt(
    user_prompt: str,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    reference_images: Optional[List[Image.Image]] = None,
    return_raw_output: bool = False,
    image_description: str = None,
) -> Tuple[List[str], Optional[str]]:
    """
    Enhances a user prompt into a list of structured, camera-aware prompts for image generation.
    Uses dynamic template fields and context-aware enhancement.

    Args:
        user_prompt (str): Base user input
        num_prompts (int): Number of prompts to generate
        reference_images (Optional[List[Image.Image]]): List of PIL images for reference
        return_raw_output (bool): If True, return raw GPT output for debugging
        image_description (str, optional): Description of the reference image

    Returns:
        Tuple[List[str], Optional[str]]: List of prompts and optional raw GPT output string
    """
    logger.info(f"Starting prompt enhancement for: {user_prompt}")

    # === Step 1: Determine prompt intent ===
    intent = classify_prompt_intent(user_prompt)
    logger.info(f"Inferred intent label: {intent}")

    # === Step 2: Get visual style hint from reference images ===
    style_hint = ""
    full_caption = ""
    product_description = ""
    if reference_images:
        logger.info(f"Analyzing {len(reference_images)} reference images...")
        try:
            blip_data = describe_uploaded_images(reference_images)
            style_hint = blip_data.get("style_description", "")
            full_caption = blip_data.get("full_caption", "")
            product_description = blip_data.get("product_description", "")
            logger.info(f"Image analysis results - Style: {style_hint}, Caption: {full_caption}")
        except Exception as e:
            logger.warning(f"Failed to analyze reference images: {e}")

    # === Step 3: Build Enhancement Context ===
    context = EnhancementContext(
        user_prompt=user_prompt,
        num_prompts=num_prompts,
        reference_images=reference_images,
        style_hint=style_hint,
        product_description=product_description,
        image_description=image_description,
        intent=intent
    )

    # === Step 4: Build System Message ===
    system_msg = build_dynamic_system_message(context)
    logger.debug(f"System message: {system_msg}")
    
    # === Step 5: Build User Message ===
    user_message = f"Original prompt: {user_prompt}"
    if style_hint:
        user_message += f"\nVisual reference style: {style_hint}"
    if product_description:
        user_message += f"\nReference product: {product_description}"
    logger.debug(f"User message: {user_message}")

    try:
        # === Step 6: GPT Completion Call ===
        logger.info("Calling GPT for prompt enhancement...")
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_message},
            ],
            temperature=0.7,
            max_tokens=1800,
        )

        raw_output = response.choices[0].message.content.strip()
        logger.info(f"Raw GPT Output: {raw_output}")

        # === Step 7: Parse and Clean Output ===
        if raw_output.startswith("[") and raw_output.endswith("]"):
            try:
                parsed = ast.literal_eval(raw_output)
                if isinstance(parsed, list):
                    raw_output = "\n".join(parsed)
                    logger.debug("Parsed list to lines")
            except Exception as e:
                logger.warning(f"Failed to parse GPT output as list: {e}")

        # === Step 8: Extract Valid Prompts ===
        prompts = extract_valid_prompts(
            raw_output, num_prompts, fallback_prompt=user_prompt
        )

        logger.info(f"Generated {len(prompts)} valid prompts: {prompts}")
        return (prompts, raw_output) if return_raw_output else (prompts, None)

    except Exception as e:
        logger.error(f"GPT prompt enhancement failed: {str(e)}", exc_info=True)
        return ([user_prompt] * num_prompts, None)

def enhance_prompt_with_similars(user_prompt: str, similar_prompts: List[str], num_variations: int = 5) -> List[str]:
    """
    Leverage similar prompts to enrich context and guide GPT into generating more relevant, diverse, and higher-quality enhancements.

    Args:
        user_prompt (str): The user's base prompt
        similar_prompts (List[str]): List of similar prompts retrieved from Milvus or history
        num_variations (int): Number of enhanced prompts to generate

    Returns:
        List[str]: List of enhanced, structured prompts
    """
    
    reference_block = "The following prompts were previously successful:\n"
    reference_block += "\n".join([f"{i+1}. {p.strip()}" for i, p in enumerate(similar_prompts)])

    # === [IMPROVED] Add explicit goal for GPT in user message ===
    guidance = (
        f"\n\nNow, based on the user prompt and these reference examples, generate {num_variations} enhanced prompts.\n"
        f"Each prompt must be a fluent, single sentence under 120 words, adhering to the visual storytelling structure defined by the assistant.\n"
        f"User prompt: {user_prompt.strip()}"
    )

    combined = f"{reference_block}\n{guidance}"

    
    prompts, _ = enhance_prompt_with_chatgpt(
        combined,
        num_prompts=num_variations,
        reference_images=None,
        return_raw_output=False,
        image_description=None
    )
    return prompts

class PromptEnhancer:
    """Main class for prompt enhancement operations."""
    
    def __init__(self, config: Any = DEFAULT_PROMPT_CONFIG):
        """
        Initialize the prompt enhancer.
        
        Args:
            config: Configuration for prompt enhancement
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.config = config

    def clean_repetitive_phrases(self, text: str) -> str:
        """
        Remove consecutive repeated words.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        return re.sub(r'\b(\w+)(?: \1\b)+', r'\1', text, flags=re.IGNORECASE)

    def enhance_prompt(
        self,
        prompt: str,
        num_variations: int = 1,
        style_hint: str = None,
        reference_images: Optional[List[Image.Image]] = None,
        image_description: str = None,
    ) -> list:
        """
        Enhance the given prompt using the structured template logic.

        Args:
            prompt (str): The original prompt to enhance
            num_variations (int): Number of variations to generate
            style_hint (str): Optional style hint
            reference_images (List[Image.Image], optional): List of reference images
            image_description (str, optional): Description of the reference image

        Returns:
            list: The enhanced prompts
        """
        if image_description:
            image_description = self.clean_repetitive_phrases(image_description)
            
        prompts, _ = enhance_prompt_with_chatgpt(
            user_prompt=prompt,
            num_prompts=num_variations,
            reference_images=reference_images,
            return_raw_output=False,
            image_description=image_description,
        )
        return prompts

    def batch_enhance_prompts(
        self, prompts: List[str], reference_images: Optional[List[Image.Image]] = None
    ) -> List[str]:
        """
        Enhance multiple prompts in batch.

        Args:
            prompts (List[str]): List of original prompts to enhance
            reference_images (List[Image.Image], optional): List of reference images

        Returns:
            List[str]: List of enhanced prompts
        """
        return [
            self.enhance_prompt(prompt, reference_images=reference_images)
            for prompt in prompts
        ]
