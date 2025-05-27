import os
import re
from typing import List, Dict, Optional

import torch
import openai
from PIL import Image

from .model_manager import get_model_manager

# === Constants ===
VISION_DEVICE = os.getenv("VISION_DEVICE", "cpu")
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", 120))
DEFAULT_VISION_PROMPT = "a photograph of"

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


def analyze_caption_with_gpt(caption: str) -> Dict[str, str]:
    """
    Analyze a caption using GPT to extract structured information.

    Args:
        caption (str): Image caption to analyze

    Returns:
        Dict[str, str]: Structured information about the image
    """
    print(f"\n🎯 [DEBUG] Analyzing caption with GPT: {caption[:100]}...")

    system_message = (
        "You are an image caption analyzer for a commercial image generation system.\n"
        "Given a caption, return three things:\n"
        "1. The main product or object described\n"
        "2. The visual style or medium (photo, sketch, digital art, etc.)\n"
        "3. The inferred intent (e.g., product-ad, awareness, creative-art, promotion)\n"
        "Only return a JSON dictionary with keys: product, style, intent."
    )

    user_message = f"Image caption: {caption}"
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=400,
        )
        content = response.choices[0].message.content.strip()
        parsed = eval(content) if content.startswith("{") else {}
        result = parsed if isinstance(parsed, dict) else {}

        print("✅ [DEBUG] GPT analysis results:")
        for key, value in result.items():
            print(f"  - {key}: {value}")

        return result
    except Exception as e:
        print(f"❌ [GPT ERROR] Failed to analyze caption: {e}")
        return {}


class VisionDescriber:
    """
    Describes images using a vision-language model.
    """

    def __init__(
        self, device: str = VISION_DEVICE, max_tokens: int = VISION_MAX_TOKENS
    ):
        """
        Initialize the vision describer.

        Args:
            device (str): Device to run the model on
            max_tokens (int): Maximum tokens for generation
        """
        self.device = device
        self.max_tokens = max_tokens
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load the vision model and processor."""
        if self.processor is None or self.model is None:
            print(f"🚀 [VISION] Loading model on {self.device}")
            try:
                model_manager = get_model_manager()
                self.model, self.processor = model_manager.load_model(
                    device=self.device
                )
                print("✅ [VISION] Model loaded successfully")
            except Exception as e:
                print(f"❌ [VISION ERROR] Failed to load model: {e}")
                raise

    def describe_image(
        self, image: Image.Image, prompt: str = DEFAULT_VISION_PROMPT
    ) -> str:
        """
        Describe an image using the vision model.

        Args:
            image (Image.Image): Image to describe
            prompt (str): Prompt for the model

        Returns:
            str: Generated description
        """
        try:
            print(f"\n🎯 [DEBUG] Describing image with prompt: {prompt[:100]}...")

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

            print(f"✅ [DEBUG] Generated caption: {cleaned}")
            return cleaned
        except Exception as e:
            print(f"❌ [VISION ERROR] Failed to describe image: {e}")
            return ""


# === Singleton ===
_describer = VisionDescriber()


def describe_uploaded_images(
    images: List[Image.Image], prompt_override: str = ""
) -> Dict[str, str]:
    """
    Describe multiple uploaded images and analyze their content.

    Args:
        images (List[Image.Image]): List of images to describe
        prompt_override (str): Optional custom prompt

    Returns:
        Dict[str, str]: Structured information about the images
    """
    if not images:
        print("⚠️ [DEBUG] No images provided")
        return {
            "style_description": "",
            "full_caption": "",
            "product_description": "",
            "intent_description": "",
        }

    print(f"\n🎯 [DEBUG] Processing {len(images)} images")
    prompt = prompt_override if prompt_override else DEFAULT_VISION_PROMPT
    captions = []

    for i, img in enumerate(images, 1):
        print(f"\n📸 [DEBUG] Processing image {i}/{len(images)}")
        caption = _describer.describe_image(img, prompt)
        if caption:
            captions.append(caption)

    combined_caption = "; ".join(sorted(set(captions)))
    print(f"\n📝 [DEBUG] Combined caption: {combined_caption}")

    gpt_insights = analyze_caption_with_gpt(combined_caption)

    result = {
        "full_caption": combined_caption,
        "style_description": gpt_insights.get("style", ""),
        "product_description": gpt_insights.get("product", ""),
        "intent_description": gpt_insights.get("intent", ""),
    }

    print("\n✅ [DEBUG] Final results:")
    for key, value in result.items():
        print(f"  - {key}: {value}")

    return result
