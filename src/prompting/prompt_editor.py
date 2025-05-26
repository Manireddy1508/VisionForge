from typing import List, Optional, Tuple
from PIL import Image

from prompting.constants import DEFAULT_NUM_PROMPTS
from prompting.prompt_enhancer import enhance_prompt_with_chatgpt


def generate_editable_prompts(
    user_prompt: str,
    reference_images: Optional[List[Image.Image]] = None,
    num_outputs: int = DEFAULT_NUM_PROMPTS,
    style_hint: Optional[str] = None,
) -> List[str]:
    """
    Generates enhanced prompts and pads them for editable display in the UI.
    This function serves as a bridge between the prompt enhancement system and the UI.

    Args:
        user_prompt (str): The original user prompt.
        reference_images (List[PIL.Image], optional): List of uploaded reference images.
        num_outputs (int): Number of prompts to generate (user selected).
        style_hint (str, optional): Optional style hint for prompt enhancement.

    Returns:
        List[str]: Padded list of enhanced prompts (length = DEFAULT_NUM_PROMPTS).
    """
    print("\n🎯 [DEBUG] Starting prompt generation in editor...")
    print(f"📌 [DEBUG] Base prompt: {user_prompt}")
    print(f"📌 [DEBUG] Number of outputs requested: {num_outputs}")
    print(
        f"📌 [DEBUG] Number of reference images: {len(reference_images) if reference_images else 0}"
    )
    if style_hint:
        print(f"📌 [DEBUG] Style hint: {style_hint}")

    try:
        prompts, raw_output = enhance_prompt_with_chatgpt(
            user_prompt=user_prompt,
            num_prompts=num_outputs,
            reference_images=reference_images,
            return_raw_output=True,  # Get raw output for debugging
        )

        if not isinstance(prompts, list):
            prompts = [str(prompts)]

        print("\n📝 [DEBUG] Prompt Preview Output:")
        for i, p in enumerate(prompts):
            print(f"  [{i+1}] {p}")

        # Ensure output list is always DEFAULT_NUM_PROMPTS long
        padded_prompts = prompts + [""] * (DEFAULT_NUM_PROMPTS - len(prompts))
        print(
            f"✅ [DEBUG] Generated {len(prompts)} prompts, padded to {DEFAULT_NUM_PROMPTS}"
        )

        return padded_prompts

    except Exception as e:
        print(f"❌ [ERROR] Failed to generate prompts: {str(e)}")
        # Return empty prompts on error
        return [""] * DEFAULT_NUM_PROMPTS


def batch_generate_prompts(
    prompts: List[str],
    reference_images: Optional[List[Image.Image]] = None,
    num_outputs: int = DEFAULT_NUM_PROMPTS,
    style_hint: Optional[str] = None,
) -> List[List[str]]:
    """
    Batch generate editable prompts for multiple input prompts.

    Args:
        prompts (List[str]): List of original user prompts.
        reference_images (List[PIL.Image], optional): List of uploaded reference images.
        num_outputs (int): Number of prompts to generate per input.
        style_hint (str, optional): Optional style hint for prompt enhancement.

    Returns:
        List[List[str]]: List of padded prompt lists.
    """
    print(f"\n🔄 [DEBUG] Starting batch prompt generation for {len(prompts)} inputs")
    return [
        generate_editable_prompts(
            user_prompt=prompt,
            reference_images=reference_images,
            num_outputs=num_outputs,
            style_hint=style_hint,
        )
        for prompt in prompts
    ]
