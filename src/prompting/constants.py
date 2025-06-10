"""
Constants and configuration for the prompt enhancement system.
Provides dynamic template structures and configuration management.
"""

from typing import Dict, List, Optional, TypedDict
from dataclasses import dataclass
from enum import Enum, auto

# === Template Field Definitions ===
class TemplateFieldType(Enum):
    REQUIRED = auto()
    OPTIONAL = auto()
    CONTEXTUAL = auto()

@dataclass
class TemplateField:
    name: str
    field_type: TemplateFieldType
    description: str
    constraints: Optional[List[str]] = None
    examples: Optional[List[str]] = None

# === Dynamic Template Structure ===
TEMPLATE_FIELDS = {
    "art_medium": TemplateField(
        name="art medium",
        field_type=TemplateFieldType.REQUIRED,
        description="The artistic medium or style of the image",
        constraints=["Must be appropriate for the context", "Should enhance the main subject"]
    ),
    "main_object": TemplateField(
        name="main object or objective",
        field_type=TemplateFieldType.REQUIRED,
        description="The primary subject or focus of the image",
        constraints=["Must be clearly defined", "Should be the central element"]
    ),
    "attribute": TemplateField(
        name="attribute",
        field_type=TemplateFieldType.REQUIRED,
        description="Key characteristics or qualities of the main subject",
        constraints=["Must be relevant to the subject", "Should enhance understanding"]
    ),
    "expression": TemplateField(
        name="expression",
        field_type=TemplateFieldType.REQUIRED,
        description="The emotional or visual tone of the image",
        constraints=["Must align with intent", "Should be consistent"]
    ),
    "key_light": TemplateField(
        name="key light",
        field_type=TemplateFieldType.REQUIRED,
        description="The primary lighting setup or mood",
        constraints=["Must be physically plausible", "Should enhance the subject"]
    ),
    "detailing": TemplateField(
        name="detailing",
        field_type=TemplateFieldType.REQUIRED,
        description="Specific details or features to emphasize",
        constraints=["Must be relevant", "Should add value"]
    ),
    "camera_type": TemplateField(
        name="camera type",
        field_type=TemplateFieldType.OPTIONAL,
        description="The type of camera or perspective",
        constraints=["Must be appropriate for the context", "Should enhance realism"]
    ),
    "camera_angle": TemplateField(
        name="camera angle",
        field_type=TemplateFieldType.OPTIONAL,
        description="The viewing angle or perspective",
        constraints=["Must be physically possible", "Should enhance composition"]
    ),
    "artistic_technique": TemplateField(
        name="artistic technique",
        field_type=TemplateFieldType.OPTIONAL,
        description="Specific artistic methods or styles",
        constraints=["Must be appropriate for the medium", "Should enhance the result"]
    )
}

DEFAULT_NUM_PROMPTS = 5
MAX_PROMPT_LENGTH = 150# === Backwards Compatibility Constants ===
MIN_WORD_COUNT = 8
ENFORCE_SINGLE_SENTENCE = True
STRIP_QUOTES = True
SUPPRESS_NUMBERING = True
IGNORE_EXPLICIT_COLORS = True
REQUIRE_REALISTIC_SCALE = True
ALLOW_CUSTOM_INTENTS = True
ENABLE_ARTISTIC_TECHNIQUES = True
ENABLE_DYNAMIC_COMPOSITION = True
ENABLE_STYLE_TRANSFER = True

# === Configuration Management ===
@dataclass
class PromptConfig:
    max_length: int = 150
    default_num_prompts: int = 5
    min_word_count: int = 8
    enforce_single_sentence: bool = True
    strip_quotes: bool = True
    suppress_numbering: bool = True
    ignore_explicit_colors: bool = True
    require_realistic_scale: bool = True
    allow_custom_intents: bool = True
    enable_artistic_techniques: bool = True
    enable_dynamic_composition: bool = True
    enable_style_transfer: bool = True

# === System Configuration ===
class SystemConfig(TypedDict):
    roles: Dict[str, str]
    style_keys: Dict[str, str]
    intent_instructions: Dict[str, str]

# === Default Configurations ===
DEFAULT_PROMPT_CONFIG = PromptConfig()

DEFAULT_SYSTEM_CONFIG = SystemConfig(
    roles={
        "prompt_enhancer": "You are a prompt enhancement assistant for AI image generation.",
        "intent_classifier": "You are an intent classification engine for AI image generation.",
        "image_describer": "You are an image analysis module for AI image generation."
    },
    style_keys={
        "style_description": "style_description",
        "full_caption": "full_caption"
    },
    intent_instructions={
    "product-ad": (
        "Think like a professional photographer. Focus on commercial visual "
        "storytelling that highlights the product's usage, form, and emotional "
        "appeal. Include natural camera decisions (angle or type) when they "
        "enhance realism or brand perception."
    ),
    "service-promotion": (
        "Imagine you're capturing the real-life moment a service is being used. "
        "Emphasize tone, setting, and clarity. You may use cinematic framing or "
        "lifestyle-oriented composition where helpful."
    ),
    "public-awareness": (
        "Use symbolic or emotional imagery to visually communicate the importance "
        "of a cause or campaign. Consider using artistic techniques that enhance "
        "the emotional impact."
    ),
    "brand-storytelling": (
        "Craft lifestyle-driven prompts that reflect the brand's values. When "
        "appropriate, you may include a camera perspective that emphasizes mood, "
        "composition, or the viewer's relationship to the scene."
    ),
    "artistic-expression": (
        "Focus on creative and artistic elements. Emphasize visual style, "
        "composition, and emotional impact. Feel free to incorporate specific "
        "artistic techniques and mediums."
    ),
    "social-trend": (
        "Create prompts that align with current social media trends and viral "
        "content styles. Consider popular visual aesthetics and contemporary "
        "artistic movements."
    ),
    "educational-content": (
        "Design clear, informative visuals that effectively communicate "
        "educational concepts. Use appropriate artistic techniques to enhance "
        "clarity and engagement."
    ),
    "campaign-launch": (
        "Create impactful, attention-grabbing prompts suitable for marketing "
        "campaign launches. Incorporate dynamic composition and compelling "
        "visual elements."
    ),
    "experimental-style": (
        "Push creative boundaries with unique and innovative visual approaches. "
        "Combine different artistic techniques and mediums for novel effects."
        )
}
)

# === Backwards Compatibility ===
# These maintain compatibility with existing code while using the new structure
PROMPT_PREFIX = "It's very important that [main object or objective] isn't modified"
PROMPT_SUFFIX = (
    "and take all the time you needed as it is very important to achieve the "
    "best possible result"
)

TEMPLATE_FORMAT = (
    "prefix [art medium] [main object or objective] [attribute] [expression] "
    "[key light] [detailing] [optional: camera type] [optional: camera angle] "
    "[optional: artistic technique] suffix"
)

REQUIRED_ELEMENTS = [field.name for field in TEMPLATE_FIELDS.values() 
                    if field.field_type == TemplateFieldType.REQUIRED]

OPTIONAL_ELEMENTS = [field.name for field in TEMPLATE_FIELDS.values() 
                    if field.field_type == TemplateFieldType.OPTIONAL]

# === Export Configuration ===
__all__ = [
    'TEMPLATE_FIELDS',
    'PromptConfig',
    'SystemConfig',
    'DEFAULT_PROMPT_CONFIG',
    'DEFAULT_SYSTEM_CONFIG',
    'PROMPT_PREFIX',
    'PROMPT_SUFFIX',
    'TEMPLATE_FORMAT',
    'REQUIRED_ELEMENTS',
    'OPTIONAL_ELEMENTS'
]

