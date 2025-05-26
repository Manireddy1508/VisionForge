import os
import openai
import ast
from typing import List, Optional, Tuple
from PIL import Image
from openai import OpenAI

from prompting.constants import DEFAULT_NUM_PROMPTS
from prompting.prompt_router import classify_prompt_intent
from prompting.image_describer import describe_uploaded_images
from prompting.prompt_templates import build_system_message
from prompting.prompt_utils import extract_valid_prompts

# === OpenAI Client ===
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def enhance_prompt_with_chatgpt(
    user_prompt: str,
    num_prompts: int = DEFAULT_NUM_PROMPTS,
    reference_images: Optional[List[Image.Image]] = None,
    return_raw_output: bool = False,
) -> Tuple[List[str], Optional[str]]:
    """
    Enhances a user prompt into a list of structured, camera-aware prompts for image generation.
    Now includes BLIP image analysis for reference images.

    Args:
        user_prompt (str): Base user input.
        num_prompts (int): Number of prompts to generate.
        reference_images (Optional[List[Image.Image]]): List of PIL images for reference.
        return_raw_output (bool): If True, return raw GPT output for debugging.

    Returns:
        Tuple[List[str], Optional[str]]: List of prompts and optional raw GPT output string.
    """
    print("\n🎯 [DEBUG] Starting prompt enhancement...")

    # === Step 1: Determine prompt intent ===
    intent = classify_prompt_intent(user_prompt)
    print(f"📌 [DEBUG] Inferred intent label: {intent}")

    # === Step 2: Get visual style hint from reference images ===
    style_hint = ""
    full_caption = ""
    if reference_images:
        print(f"📸 [DEBUG] Analyzing {len(reference_images)} reference images...")
        try:
            blip_data = describe_uploaded_images(reference_images)
            style_hint = blip_data.get("style_description", "")
            full_caption = blip_data.get("full_caption", "")
        except Exception as e:
            print(f"⚠️ [WARN] Failed to analyze reference images: {e}")

    print(f"\n📥 [DEBUG] User Prompt: {user_prompt}")
    if full_caption:
        print(f"🖼️ [DEBUG] BLIP Caption: {full_caption}")
    if style_hint:
        print(f"🎨 [DEBUG] Style Hint: {style_hint}")

    # === Step 3: GPT System + User Message Composition ===
    system_msg = build_system_message(intent, num_prompts, style_hint)
    user_message = f"Original prompt: {user_prompt}"
    if style_hint:
        user_message += f"\nVisual reference style: {style_hint}"

    try:
        # === Step 4: GPT Completion Call ===
        print("\n🤖 [DEBUG] Calling GPT for prompt enhancement...")
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
        print("\n📝 [DEBUG] Raw GPT Output:")
        print(raw_output)

        # === Fix if GPT returns stringified list ===
        if raw_output.startswith("[") and raw_output.endswith("]"):
            try:
                parsed = ast.literal_eval(raw_output)
                if isinstance(parsed, list):
                    raw_output = "\n".join(parsed)
                    print("\n🧠 [DEBUG] Parsed list to lines:")
                    for i, p in enumerate(parsed):
                        print(f"[{i+1}] {p}")
            except Exception as e:
                print(f"⚠️ [WARN] Failed to parse GPT output as list: {e}")

        # === Step 5: Extract Valid Prompts ===
        prompts = extract_valid_prompts(
            raw_output, num_prompts, fallback_prompt=user_prompt
        )

        print("\n🔧 [DEBUG] Final Enhanced Prompts:")
        for idx, p in enumerate(prompts):
            print(f"[{idx+1}] {p}")
        print("--------------------------------------------------\n")

        return (prompts, raw_output) if return_raw_output else (prompts, None)

    except Exception as e:
        print(f"❌ [ERROR] GPT prompt enhancement failed: {e}")
        return ([user_prompt] * num_prompts, None)


class PromptEnhancer:
    def __init__(self):
        """Initialize the prompt enhancer with OpenAI client."""
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def enhance_prompt(
        self,
        prompt: str,
        num_variations: int = 1,
        style_hint: str = None,
        reference_images: Optional[List[Image.Image]] = None,
    ) -> list:
        """
        Enhance the given prompt using the structured template logic.
        Now includes BLIP image analysis for reference images.

        Args:
            prompt (str): The original prompt to enhance
            num_variations (int): Number of variations to generate
            style_hint (str): Optional style hint
            reference_images (List[Image.Image], optional): List of reference images

        Returns:
            list: The enhanced prompts (as a list)
        """
        prompts, _ = enhance_prompt_with_chatgpt(
            user_prompt=prompt,
            num_prompts=num_variations,
            reference_images=reference_images,
            return_raw_output=False,
        )
        return prompts

    def batch_enhance_prompts(
        self, prompts: List[str], reference_images: Optional[List[Image.Image]] = None
    ) -> List[str]:
        """
        Enhance multiple prompts in batch.
        Now includes BLIP image analysis for reference images.

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
