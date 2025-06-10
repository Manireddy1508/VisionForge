"""
Prompt editing and enhancement functionality for the image generation system.
Provides structured prompt editing with context-aware enhancement.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple

from PIL import Image

from prompting.constants import (
    PromptConfig,
    DEFAULT_PROMPT_CONFIG,
)
from prompting.prompt_enhancer import (
    enhance_prompt_with_chatgpt,
    EnhancementContext,
)
from prompting.prompt_router import PromptRouter

# === Logging Setup ===
logger = logging.getLogger(__name__)

def pad_prompts(prompts: List[str], target_length: int) -> List[str]:
    """
    Pad a list of prompts to a target length with empty strings.
    
    Args:
        prompts (List[str]): List of prompts to pad
        target_length (int): Target length for the output list
        
    Returns:
        List[str]: Padded list of prompts
    """
    return prompts + [""] * (target_length - len(prompts))

def generate_editable_prompts(
    context: EnhancementContext,
) -> List[str]:
    """
    Generates enhanced prompts and pads them for editable display in the UI.
    This function serves as a bridge between the prompt enhancement system and the UI.

    Args:
        context (EnhancementContext): Context containing all relevant information

    Returns:
        List[str]: Padded list of enhanced prompts (length = context.config.default_num_prompts).
    """
    logger.info("Starting prompt generation in editor...")
    logger.debug(f"Base prompt: {context.user_prompt}")
    logger.debug(f"Number of outputs requested: {context.num_prompts}")
    logger.debug(
        f"Number of reference images: {len(context.reference_images) if context.reference_images else 0}"
    )
    if context.style_hint:
        logger.debug(f"Style hint: {context.style_hint}")

    try:
        prompts, raw_output = enhance_prompt_with_chatgpt(
            user_prompt=context.user_prompt,
            num_prompts=context.num_prompts,
            reference_images=context.reference_images,
            return_raw_output=True,  # Get raw output for debugging
        )

        if not isinstance(prompts, list):
            prompts = [str(prompts)]

        logger.debug("Prompt Preview Output:")
        for i, p in enumerate(prompts):
            logger.debug(f"  [{i+1}] {p}")

        # Ensure output list is always the default length
        padded_prompts = pad_prompts(prompts, context.config.default_num_prompts)
        logger.info(
            f"Generated {len(prompts)} prompts, padded to {context.config.default_num_prompts}"
        )

        return padded_prompts

    except Exception as e:
        logger.error(f"Failed to generate prompts: {e}")
        # Return empty prompts on error
        return [""] * context.config.default_num_prompts

def batch_generate_prompts(
    contexts: List[EnhancementContext],
) -> List[List[str]]:
    """
    Batch generate editable prompts for multiple input contexts.

    Args:
        contexts (List[EnhancementContext]): List of enhancement contexts

    Returns:
        List[List[str]]: List of padded prompt lists
    """
    logger.info(f"Starting batch prompt generation for {len(contexts)} inputs")
    return [generate_editable_prompts(context) for context in contexts]

class PromptEditor:
    """Handles prompt editing and enhancement functionality."""
    
    def __init__(self, config: PromptConfig = DEFAULT_PROMPT_CONFIG):
        """
        Initialize the prompt editor.
        
        Args:
            config (PromptConfig): Configuration for prompt editing
        """
        self.prompt_router = PromptRouter(config)
        self.config = config
        logger.info(f"Initialized PromptEditor with config: {config}")
    
    def edit_prompt(
        self,
        context: EnhancementContext,
    ) -> List[str]:
        """
        Edit and enhance a prompt using the full pipeline.
        
        Args:
            context (EnhancementContext): Context containing all relevant information
            
        Returns:
            List[str]: List of enhanced prompts
        """
        try:
            # First route the prompt through the router
            routed_prompt = self.prompt_router.route_prompt(context.user_prompt)
            
            # Create new context with routed prompt
            routed_context = EnhancementContext(
                user_prompt=routed_prompt,
                num_prompts=context.num_prompts,
                reference_images=context.reference_images,
                style_hint=context.style_hint,
                product_description=context.product_description,
                image_description=context.image_description,
                intent=context.intent,
                config=context.config
            )
            
            # Generate enhanced prompts
            return generate_editable_prompts(routed_context)
            
        except Exception as e:
            logger.error(f"Error editing prompt: {e}")
            return [""] * context.num_prompts
    
    def batch_edit_prompts(
        self,
        contexts: List[EnhancementContext],
    ) -> List[List[str]]:
        """
        Edit multiple prompts in batch using the full pipeline.
        
        Args:
            contexts (List[EnhancementContext]): List of enhancement contexts
            
        Returns:
            List[List[str]]: List of enhanced prompt lists
        """
        try:
            return [self.edit_prompt(context) for context in contexts]
        except Exception as e:
            logger.error(f"Error in batch editing prompts: {e}")
            return [[""] * context.num_prompts for context in contexts]
