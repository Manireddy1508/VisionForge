"""
Template management and prompt building for the image generation system.
Provides dynamic template-based prompt generation and configuration.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

from prompting.constants import (
    TEMPLATE_FIELDS,
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    TEMPLATE_FORMAT,
    MAX_PROMPT_LENGTH,
    REQUIRED_ELEMENTS,
    OPTIONAL_ELEMENTS,
    ARTISTIC_TECHNIQUES,
    ART_MEDIUM_DEFINITION,
    CAMERA_TYPES,
    CAMERA_ANGLES,
    PromptConfig,
    DEFAULT_PROMPT_CONFIG,
    DEFAULT_SYSTEM_CONFIG,
)

# === Logging Setup ===
logger = logging.getLogger(__name__)

@dataclass
class TemplateContext:
    """Context for template-based prompt generation."""
    intent: str
    num_prompts: int
    style_hint: str = ""
    product_description: str = ""
    image_description: str = ""
    config: PromptConfig = field(default_factory=lambda: DEFAULT_PROMPT_CONFIG)

@dataclass
class PromptTemplate:
    """Represents a structured prompt template."""
    fields: Dict[str, str]
    prefix: str = PROMPT_PREFIX
    suffix: str = PROMPT_SUFFIX
    is_valid: bool = True
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    def to_string(self) -> str:
        """Convert template to a formatted prompt string."""
        # Build the main content
        content = []
        for field_name in TEMPLATE_FIELDS:
            if field_name in self.fields:
                content.append(f"[{self.fields[field_name]}]")
        
        # Combine with prefix and suffix
        return f"{self.prefix} {' '.join(content)} {self.suffix}"

class PromptTemplateBuilder:
    """
    Builds and validates prompt templates using dynamic field definitions.
    Uses template fields from constants.py for structure and validation.
    """

    def __init__(self, config: PromptConfig = DEFAULT_PROMPT_CONFIG):
        """
        Initialize the template builder.

    Args:
            config: Configuration for template building
        """
        self.config = config
        logger.info(f"Initialized PromptTemplateBuilder with config: {config}")

    def build_template(self, fields: Dict[str, str]) -> PromptTemplate:
        """
        Build a prompt template from field values.

    Args:
            fields (Dict[str, str]): Dictionary of field names to values

    Returns:
            PromptTemplate: The built template
        """
        template = PromptTemplate(fields=fields)
        
        # Validate required fields
        missing_fields = set(REQUIRED_ELEMENTS) - set(fields.keys())
        if missing_fields:
            template.is_valid = False
            template.warnings.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate field constraints
        for field_name, value in fields.items():
            field = TEMPLATE_FIELDS.get(field_name)
            if field and field.constraints:
                for constraint in field.constraints:
                    if not self._validate_constraint(value, constraint):
                        template.warnings.append(
                            f"Field '{field_name}' violates constraint: {constraint}"
                        )
                        template.is_valid = False
        
        return template

    def _validate_constraint(self, value: str, constraint: str) -> bool:
        """Validate a field value against its constraint."""
        if "must be" in constraint.lower():
            return bool(value.strip())
        return True

    def build_system_message(self, context: TemplateContext) -> str:
        """
        Build a system message for GPT based on template context.

        Args:
            context: TemplateContext containing all relevant information

        Returns:
            str: The complete system message
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
        
        # Build base message
        message = f"""
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
            message += f"\nVisual style reference: {context.style_hint}"
        if context.product_description:
            message += f"\nProduct context: {context.product_description}"
        if context.image_description:
            message += f"\nAdditional image context: {context.image_description}"

        # Add intent-specific instructions
        intent_instruction = DEFAULT_SYSTEM_CONFIG["intent_instructions"].get(
            context.intent,
            f"Use your full understanding of visual storytelling to create prompts "
            f"that suit this intent: '{context.intent}'. Adapt the tone, framing, "
            f"and technical language as needed."
        )
        message += f"\n\nIntent-specific instructions:\n{intent_instruction}"

        return message.strip()

    def reconstruct_prompt(self, field_values: Dict[str, str]) -> str:
        """
        Reconstruct a prompt from extracted field values.

        Args:
            field_values (Dict[str, str]): Dictionary of field names to values

        Returns:
            str: The reconstructed prompt
        """
        template = self.build_template(field_values)
        return template.to_string()

def get_prompt_template(
    intent: str,
    num_prompts: int = 1,
    style_hint: Optional[str] = None,
    reference_images: Optional[List] = None,
    config: PromptConfig = DEFAULT_PROMPT_CONFIG,
) -> str:
    """
    Get a complete prompt template based on intent and context.
    
    Args:
        intent (str): The intent label
        num_prompts (int): Number of prompts to generate
        style_hint (str, optional): Optional style hint from reference images
        reference_images (List, optional): List of reference images
        config (PromptConfig): Configuration for template building
        
    Returns:
        str: The complete prompt template
    """
    try:
        # Create template context
        context = TemplateContext(
            intent=intent,
            num_prompts=num_prompts,
            style_hint=style_hint or "",
            config=config
        )
        
        # Build template using builder
        builder = PromptTemplateBuilder(config=config)
        system_message = builder.build_system_message(context)
        
        # Add reference image count
        if reference_images:
            system_message += f"\n\nReference Images: {len(reference_images)}"
        
        return system_message.strip()
        
    except Exception as e:
        logger.error(f"Error generating prompt template: {e}")
        # Return a basic template as fallback
        return f"""
You are a prompt enhancement assistant for AI image generation.

Your task is to transform a short user input into {num_prompts} fully structured, highly descriptive prompts suitable for AI image generation.

Each prompt must follow this format:
{" ".join(f"[{field.name}]" for field in TEMPLATE_FIELDS.values())}

Prefix: "{PROMPT_PREFIX}"
Suffix: "{PROMPT_SUFFIX}"

Prompt constraints:
- Must be under {config.max_length} words
- Must include all of: {", ".join(REQUIRED_ELEMENTS)}
- Optional when helpful: {", ".join(OPTIONAL_ELEMENTS)}
- Must be a single sentence (no bullet points, no numbering)
""".strip()
