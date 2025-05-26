# === Prompt Template ===
PROMPT_PREFIX = (
    "It's very important that [main object or objective] isn't modified"
)

PROMPT_SUFFIX = (
    "and take all the time you needed as it is very important to achieve the best possible result"
)

TEMPLATE_FORMAT = (
    "prefix [art medium] [main object or objective] [attribute] [expression] "
    "[key light] [detailing] [optional: camera type] [optional: camera angle] [optional: artistic technique] suffix"
)

# === Basic Settings ===
MAX_PROMPT_LENGTH = 150
DEFAULT_NUM_PROMPTS = 5
MIN_WORD_COUNT = 8

# === Required Prompt Elements ===
REQUIRED_ELEMENTS = [
    "art medium",
    "main object",
    "attribute",
    "expression",
    "key light",
    "detailing"
]

# === Optional Prompt Enhancements ===
OPTIONAL_ELEMENTS = [
    "camera type",   # e.g., DSLR, top-view drone, 35mm film, anime-style, surveillance
    "camera angle",  # e.g., over-the-shoulder, bird's eye, wide shot, macro close-up
    "artistic technique"  # e.g., chiaroscuro, impasto, pointillism, etc.
]

# === Camera Settings ===
CAMERA_TYPES = [
    "DSLR", "mirrorless", "drone", "35mm film", "anime-style", 
    "cinematic", "surveillance", "GoPro", "smartphone", "medium format",
    "large format", "instant camera", "toy camera", "infrared", "thermal"
]

CAMERA_ANGLES = [
    "over-the-shoulder", "low-angle", "bird's eye", "macro close-up",
    "wide shot", "top-down", "first-person", "side profile", "dutch angle",
    "aerial view", "worm's eye", "canted angle", "point-of-view", "establishing shot"
]

# === Artistic Techniques ===
ARTISTIC_TECHNIQUES = [
    "chiaroscuro", "impasto", "pointillism", "sfumato", "glazing",
    "wet-on-wet", "dry brush", "scumbling", "underpainting", "alla prima",
    "grisaille", "tenebrism", "cross-hatching", "stippling", "sgraffito"
]

# === Art Mediums ===
ART_MEDIUMS = [
    "oil painting", "watercolor", "acrylic", "digital art", "photography",
    "charcoal", "pastel", "ink", "mixed media", "gouache",
    "tempera", "fresco", "encaustic", "collage", "etching"
]

# === System Roles ===
SYSTEM_ROLES = {
    "prompt_enhancer": "You are a prompt enhancement assistant for Flux Pro.",
    "intent_classifier": "You are an intent classification engine for Flux Pro.",
    "image_describer": "You are an image analysis module for Flux Pro.",
}

# === Prompt Behavior Settings ===
PROMPT_SETTINGS = {
    "enforce_single_sentence": True,
    "strip_quotes": True,
    "suppress_numbering": True,
    "ignore_explicit_colors": True,
    "require_realistic_scale": True,
    "allow_custom_intents": True,
    "enable_artistic_techniques": True,
    "enable_dynamic_composition": True,
    "enable_style_transfer": True
}

# === Style and Caption Keys ===
STYLE_HINT_KEY = "style_description"
CAPTION_KEY = "full_caption"

# === Intent Instructions ===
INTENT_INSTRUCTIONS = {
    "product-ad": (
        "Think like a professional photographer. Focus on commercial visual storytelling that highlights the product's usage, "
        "form, and emotional appeal. Include natural camera decisions (angle or type) when they enhance realism or brand perception."
    ),
    "service-promotion": (
        "Imagine you're capturing the real-life moment a service is being used. Emphasize tone, setting, and clarity. "
        "You may use cinematic framing or lifestyle-oriented composition where helpful."
    ),
    "public-awareness": (
        "Use symbolic or emotional imagery to visually communicate the importance of a cause or campaign. "
        "Consider using artistic techniques that enhance the emotional impact."
    ),
    "brand-storytelling": (
        "Craft lifestyle-driven prompts that reflect the brand's values. When appropriate, you may include a camera perspective "
        "that emphasizes mood, composition, or the viewer's relationship to the scene."
    ),
    "artistic-expression": (
        "Focus on creative and artistic elements. Emphasize visual style, composition, and emotional impact. "
        "Feel free to incorporate specific artistic techniques and mediums."
    ),
    "social-trend": (
        "Create prompts that align with current social media trends and viral content styles. "
        "Consider popular visual aesthetics and contemporary artistic movements."
    ),
    "educational-content": (
        "Design clear, informative visuals that effectively communicate educational concepts. "
        "Use appropriate artistic techniques to enhance clarity and engagement."
    ),
    "campaign-launch": (
        "Create impactful, attention-grabbing prompts suitable for marketing campaign launches. "
        "Incorporate dynamic composition and compelling visual elements."
    ),
    "experimental-style": (
        "Push creative boundaries with unique and innovative visual approaches. "
        "Combine different artistic techniques and mediums for novel effects."
    )
} 