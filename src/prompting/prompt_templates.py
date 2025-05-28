from typing import Dict, List, Optional

from prompting.constants import (
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    TEMPLATE_FORMAT,
    MAX_PROMPT_LENGTH,
    REQUIRED_ELEMENTS,
    OPTIONAL_ELEMENTS,
    ARTISTIC_TECHNIQUES,
    ART_MEDIUMS,
    CAMERA_TYPES,
    CAMERA_ANGLES,
)

# === Optional: Custom structured template overrides per intent ===
TEMPLATE_OVERRIDES: Dict[str, str] = {
    "experimental-style": (
        "prefix [art medium] [scene] [visual tone] [expression] "
        "[composition] [artistic technique] suffix"
    ),
    "artistic-expression": (
        "prefix [art medium] [subject] [style] [technique] "
        "[composition] [lighting] suffix"
    ),
    "product-ad": (
        "prefix [art medium] [product] [setting] [mood] [lighting] "
        "[camera type] [camera angle] suffix"
    ),
    "educational-content": (
        "prefix [art medium] [subject] [style] [composition] "
        "[lighting] [technique] suffix"
    ),
}

# === Expandable intent-to-instruction mapping ===
INTENT_INSTRUCTIONS: Dict[str, str] = {
    "product-ad": (
        "Think like a professional photographer. Focus on commercial visual "
        "storytelling that highlights the product's usage, form, and emotional "
        "appeal. Include natural camera decisions (angle or type) when they "
        "enhance realism or brand perception. Consider the product's target "
        "audience and market positioning when choosing artistic style and "
        "composition."
    ),
    "service-promotion": (
        "Imagine you're capturing the real-life moment a service is being used. "
        "Emphasize tone, setting, and clarity. You may use cinematic framing or "
        "lifestyle-oriented composition where helpful. Focus on the human element "
        "and the service's impact on people's lives."
    ),
    "public-awareness": (
        "Use symbolic or emotional imagery to visually communicate the importance "
        "of a cause or campaign. Consider using artistic techniques that enhance "
        "the emotional impact. Balance between attention-grabbing visuals and "
        "clear message communication."
    ),
    "brand-storytelling": (
        "Craft lifestyle-driven prompts that reflect the brand's values. When "
        "appropriate, you may include a camera perspective that emphasizes mood, "
        "composition, or the viewer's relationship to the scene. Ensure the "
        "visual style aligns with the brand's identity and target audience."
    ),
    "artistic-expression": (
        "Focus on creative and artistic elements. Emphasize visual style, "
        "composition, and emotional impact. Feel free to incorporate specific "
        "artistic techniques and mediums. Push creative boundaries while "
        "maintaining visual coherence and impact."
    ),
    "social-trend": (
        "Create prompts that align with current social media trends and viral "
        "content styles. Consider popular visual aesthetics and contemporary "
        "artistic movements. Balance trendiness with timeless visual appeal."
    ),
    "educational-content": (
        "Design clear, informative visuals that effectively communicate "
        "educational concepts. Use appropriate artistic techniques to enhance "
        "clarity and engagement. Ensure the visual style supports the learning "
        "objectives."
    ),
    "campaign-launch": (
        "Create impactful, attention-grabbing prompts suitable for marketing "
        "campaign launches. Incorporate dynamic composition and compelling "
        "visual elements. Balance creativity with brand consistency and message "
        "clarity."
    ),
    "experimental-style": (
        "Push creative boundaries with unique and innovative visual approaches. "
        "Combine different artistic techniques and mediums for novel effects. "
        "Maintain visual coherence while exploring new possibilities."
    ),
}


def get_intent_instruction(intent: str) -> str:
    """
    Get the instruction template for a specific intent.

    Args:
        intent (str): The intent label to get instructions for

    Returns:
        str: The instruction template for the intent
    """
    return INTENT_INSTRUCTIONS.get(
        intent,
        f"Use your full understanding of visual storytelling to create prompts "
        f"that suit this intent: '{intent}'. Adapt the tone, framing, and "
        f"technical language as needed — especially when a camera type or angle "
        f"might improve realism. Consider the target audience and desired "
        f"emotional impact when choosing artistic techniques and composition.",
    )


def get_template_for_intent(intent: str) -> str:
    """
    Get the template format for a specific intent.

    Args:
        intent (str): The intent label to get template for

    Returns:
        str: The template format for the intent
    """
    return TEMPLATE_OVERRIDES.get(intent, TEMPLATE_FORMAT)


def build_system_message(intent: str, num_prompts: int, style_hint: str = "") -> str:
    """
    Build the system message for GPT based on intent and style hints.

    Args:
        intent (str): The intent label
        num_prompts (int): Number of prompts to generate
        style_hint (str): Optional style hint from reference images

    Returns:
        str: The complete system message
    """
    print(f"\n🎯 [DEBUG] Building system message for intent: {intent}")
    print(f"📌 [DEBUG] Number of prompts: {num_prompts}")
    if style_hint:
        print(f"🎨 [DEBUG] Style hint: {style_hint}")

    template_format = get_template_for_intent(intent)

    style_clause = (
        f"The visual tone, lighting, and environment must align with this style: "
        f"{style_hint}. Only override if the user explicitly requests a different "
        f"style."
        if style_hint
        else ""
    )

    camera_instruction = (
        "When the subject benefits from realistic photographic composition, infer "
        "and include specific camera types or angles. Base these decisions on "
        "what would be naturally used to capture the subject in a professional "
        "context. Do not describe camera angle generically — use descriptive, "
        "spatial, and cinematic language that aligns with the intended "
        "perspective. Avoid including any examples or sample outputs. Let your "
        "understanding of visual storytelling guide the phrasing in a way that "
        "adapts to the prompt's intent."
    )

    artistic_instruction = (
        "Consider incorporating appropriate artistic techniques that enhance the "
        "visual impact and emotional resonance. Choose techniques that complement "
        "the subject matter and intended style. Balance technical accuracy with "
        "creative expression."
    )

    system_message = f"""
You are a prompt enhancement assistant for Flux Pro.

Your task is to transform a short user input into {num_prompts} fully structured, highly descriptive prompts suitable for AI image generation.

Each prompt must follow this format:
{template_format}

Prefix: "{PROMPT_PREFIX}"
Suffix: "{PROMPT_SUFFIX}"

Prompt constraints:
- Must be under {MAX_PROMPT_LENGTH} words
- Must include all of: {", ".join(REQUIRED_ELEMENTS)}
- Optional when helpful: {", ".join(OPTIONAL_ELEMENTS)}
- Must be a single sentence (no bullet points, no numbering)
- Avoid specific color terms unless inferred from reference images
- Do not include examples or wrap prompts in quotes

All visual elements must appear with realistic physical proportions. Objects meant to be held, worn, or used must reflect true real-world size and context unless the user requests stylization or surrealism.

{style_clause}

{camera_instruction}

{artistic_instruction}

{get_intent_instruction(intent)}

Available artistic techniques: {", ".join(ARTISTIC_TECHNIQUES)}
Available art mediums: {", ".join(ART_MEDIUMS)}
Available camera types: {", ".join(CAMERA_TYPES)}
Available camera angles: {", ".join(CAMERA_ANGLES)}

Only return {num_prompts} complete prompts, one per line. Do not explain or annotate.
""".strip()

    print("\n📝 [DEBUG] Generated system message:")
    print(system_message)
    return system_message


def get_prompt_template(
    intent: str,
    num_prompts: int = 1,
    style_hint: Optional[str] = None,
    reference_images: Optional[List] = None
) -> str:
    """
    Get a complete prompt template based on intent and context.
    
    Args:
        intent (str): The intent label
        num_prompts (int): Number of prompts to generate
        style_hint (str, optional): Optional style hint from reference images
        reference_images (List, optional): List of reference images
        
    Returns:
        str: The complete prompt template
    """
    try:
        # Get the base template format
        template_format = get_template_for_intent(intent)
        
        # Get the intent-specific instruction
        instruction = get_intent_instruction(intent)
        
        # Build the system message
        system_message = build_system_message(
            intent=intent,
            num_prompts=num_prompts,
            style_hint=style_hint or ""
        )
        
        # Combine everything into a complete template
        template = f"""
{system_message}

Template Format:
{template_format}

Instructions:
{instruction}

Reference Images: {len(reference_images) if reference_images else 0}
Style Hint: {style_hint if style_hint else 'None'}
"""
        return template.strip()
        
    except Exception as e:
        print(f"Error generating prompt template: {str(e)}")
        # Return a basic template as fallback
        return f"""
You are a prompt enhancement assistant for Flux Pro.

Your task is to transform a short user input into {num_prompts} fully structured, highly descriptive prompts suitable for AI image generation.

Each prompt must follow this format:
{TEMPLATE_FORMAT}

Prefix: "{PROMPT_PREFIX}"
Suffix: "{PROMPT_SUFFIX}"

Prompt constraints:
- Must be under {MAX_PROMPT_LENGTH} words
- Must include all of: {", ".join(REQUIRED_ELEMENTS)}
- Optional when helpful: {", ".join(OPTIONAL_ELEMENTS)}
- Must be a single sentence (no bullet points, no numbering)
- Avoid specific color terms unless inferred from reference images
- Do not include examples or wrap prompts in quotes
""".strip()


def get_negative_prompt_template(
    intent: str,
    style_hint: Optional[str] = None,
    reference_images: Optional[List] = None
) -> str:
    """
    Get a template for generating negative prompts based on intent and context.
    
    Args:
        intent (str): The intent label
        style_hint (str, optional): Optional style hint from reference images
        reference_images (List, optional): List of reference images
        
    Returns:
        str: The negative prompt template
    """
    try:
        # Get intent-specific instruction
        instruction = get_intent_instruction(intent)
        
        # Build the system message
        system_message = f"""
You are a negative prompt generation assistant for Flux Pro.

Your task is to generate a negative prompt that helps avoid unwanted elements in the generated image.

Consider the following:
- Intent: {intent}
- Style Hint: {style_hint if style_hint else 'None'}
- Reference Images: {len(reference_images) if reference_images else 0}

Instructions:
{instruction}

Negative prompt constraints:
- Must be concise and specific
- Focus on elements to avoid
- Consider the intent and style
- Avoid redundant or contradictory elements
- Do not include positive elements

Example format:
"avoid [element1], [element2], [element3]"

Only return the negative prompt. Do not include explanations or examples.
""".strip()
        
        return system_message
        
    except Exception as e:
        print(f"Error generating negative prompt template: {str(e)}")
        # Return a basic template as fallback
        return """
You are a negative prompt generation assistant for Flux Pro.

Your task is to generate a negative prompt that helps avoid unwanted elements in the generated image.

Negative prompt constraints:
- Must be concise and specific
- Focus on elements to avoid
- Consider the intent and style
- Avoid redundant or contradictory elements
- Do not include positive elements

Example format:
"avoid [element1], [element2], [element3]"

Only return the negative prompt. Do not include explanations or examples.
""".strip()
