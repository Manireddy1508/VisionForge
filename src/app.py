import os
import time
import logging
from datetime import datetime
from typing import List, Optional
import gradio as gr
from dotenv import load_dotenv
from image_generator import ImageGenerator
from prompting.prompt_enhancer import PromptEnhancer

# Configure logging
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Debug logging for environment variables
logger.info("🔍 Checking environment variables...")
logger.info(f"OPENAI_IMAGE_MODEL={os.getenv('OPENAI_IMAGE_MODEL')}")
logger.info(f"OPENAI_API_KEY={'*' * 10 if os.getenv('OPENAI_API_KEY') else 'Not set'}")
logger.info(f"GOOGLE_CLOUD_PROJECT={os.getenv('GOOGLE_CLOUD_PROJECT')}")
logger.info(f"GCS_BUCKET_NAME={os.getenv('GCS_BUCKET_NAME')}")
logger.info(f"Log file: {log_file}")

# Load environment variables
load_dotenv()

# Debug logging after loading .env
logger.info("\n🔍 After loading .env file...")
logger.info(f"OPENAI_IMAGE_MODEL={os.getenv('OPENAI_IMAGE_MODEL')}")
logger.info(f"OPENAI_API_KEY={'*' * 10 if os.getenv('OPENAI_API_KEY') else 'Not set'}")
logger.info(f"GOOGLE_CLOUD_PROJECT={os.getenv('GOOGLE_CLOUD_PROJECT')}")
logger.info(f"GCS_BUCKET_NAME={os.getenv('GCS_BUCKET_NAME')}")

# Debug log for CI/CD validation
print("🔍 [CI/CD] Application initialized successfully")

# Initialize components
image_generator = ImageGenerator()
prompt_enhancer = PromptEnhancer()

# Rate limiting settings
COOLDOWN_PERIOD = 2  # seconds between requests
MAX_IMAGES = 5  # maximum number of images that can be generated at once
MAX_INPUT_IMAGES = 4  # maximum number of input images for editing


def health_check():
    """Health check endpoint for Cloud Run."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def create_demo():
    """Create the Gradio demo interface."""
    with gr.Blocks() as demo:
        gr.Markdown("## AI Image Generation Playground")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    value="a beautiful sunset over mountains",
                    placeholder="Enter your prompt here...",
                )
                with gr.Row():
                    image_uploads = [
                        gr.Image(label=f"Reference Image {i+1}", type="pil")
                        for i in range(MAX_INPUT_IMAGES)
                    ]
                with gr.Row():
                    gr.Markdown(
                        "Image size is fixed at 1024x1024 for highest quality output."
                    )
                with gr.Row():
                    num_outputs = gr.Slider(
                        1, MAX_IMAGES, 1, step=1, label="Number of Outputs"
                    )
                    seed = gr.Number(-1, label="Seed (-1 = Random)")
                with gr.Row():
                    generate_btn = gr.Button("🎯 Generate Enhanced Prompts")
                    regenerate_btn = gr.Button("🔁 Regenerate")
                    confirm_btn = gr.Button("✅ Confirm & Generate Images")
                cooldown_status = gr.Markdown("")
            with gr.Column():
                editable_prompts = []
                outputs = []
                for i in range(MAX_IMAGES):  # Maximum number of outputs
                    editable = gr.Textbox(label=f"Prompt {i+1}", lines=3)
                    image_out = gr.Image(label=f"Image {i+1}")
                    editable_prompts.append(editable)
                    outputs.append(image_out)
                    outputs.append(editable)

        def generate_prompts(*args):
            logger.info("\n🎯 Starting prompt generation...")
            prompt = args[0]
            imgs = args[1:-1]
            num_outputs = args[-1]
            logger.info(f"📌 Base prompt: {prompt}")
            logger.info(f"📌 Number of outputs requested: {num_outputs}")
            logger.info(
                f"📌 Number of reference images: "
                f"{len([img for img in imgs if img is not None])}"
            )

            uploaded_images = [img for img in imgs if img is not None]
            style_hint = None
            try:
                enhanced_prompts = prompt_enhancer.enhance_prompt(
                    prompt=prompt,
                    num_variations=num_outputs,
                    style_hint=style_hint,
                    reference_images=uploaded_images,
                )
                logger.info(
                    f"✅ Successfully generated {len(enhanced_prompts)} enhanced prompts"
                )
                return enhanced_prompts + [""] * (MAX_IMAGES - len(enhanced_prompts))
            except Exception as e:
                logger.error(f"❌ Failed to generate prompts: {str(e)}")
                raise

        def update_cooldown_status():
            return "⏳ Please wait between requests to avoid rate limits..."

        def clear_cooldown_status():
            return ""

        generate_btn.click(
            fn=generate_prompts,
            inputs=[prompt, *image_uploads, num_outputs],
            outputs=editable_prompts,
        )
        regenerate_btn.click(
            fn=generate_prompts,
            inputs=[prompt, *image_uploads, num_outputs],
            outputs=editable_prompts,
        )

        def generate_images_from_prompts(*args):
            logger.info("\n🎨 Starting image generation...")
            # args: [prompt1, prompt2, ..., prompt5, num_outputs, seed, base_prompt,
            # image1, image2, image3, image4]
            editable_prompt_count = MAX_IMAGES
            prompt_args = args[:editable_prompt_count]
            num_outputs = int(args[editable_prompt_count])
            seed = args[editable_prompt_count + 1]
            base_prompt = args[editable_prompt_count + 2]
            image_args = args[editable_prompt_count + 3 :]
            uploaded_images = [img for img in image_args if img is not None]

            logger.info(f"📌 Number of outputs requested: {num_outputs}")
            logger.info(f"📌 Seed value: {seed}")
            logger.info(f"📌 Base prompt: {base_prompt}")
            logger.info(f"📌 Number of reference images: {len(uploaded_images)}")

            selected_prompts = [p.strip() for p in prompt_args if p.strip()]
            if not selected_prompts:
                if not base_prompt or not base_prompt.strip():
                    raise ValueError(
                        "⚠️ No prompts available. Please enter a prompt or "
                        "generate enhanced prompts."
                    )
                selected_prompts = [base_prompt.strip()] * num_outputs
            if len(selected_prompts) < num_outputs:
                if len(selected_prompts) == 1:
                    selected_prompts = [selected_prompts[0]] * num_outputs
                else:
                    raise ValueError(
                        f"⚠️ You selected {num_outputs} outputs, but only provided "
                        f"{len(selected_prompts)} filled prompts."
                    )
            selected_prompts = selected_prompts[:num_outputs]
            results = []

            # Show cooldown status (pad with None for images/prompts)
            yield [None] * (MAX_IMAGES * 2) + [
                "⏳ Please wait between requests to avoid rate limits..."
            ]

            for i, final_prompt in enumerate(selected_prompts):
                try:
                    logger.info(f"\n🖼️ Generating image {i+1}/{len(selected_prompts)}")
                    logger.info(f"📌 Using prompt: {final_prompt}")

                    # If images are uploaded, use edit_images
                    if uploaded_images:
                        logger.info("📌 Using image editing mode")
                        edited_images = image_generator.edit_images(
                            images=uploaded_images,
                            prompt=final_prompt,
                            enhanced_prompt=base_prompt,  # Original prompt is enhanced
                            edited_prompt=final_prompt,  # Final prompt is edited
                            category="image_edit",
                        )
                        # Show only as many outputs as requested
                        for img_data in edited_images[:num_outputs]:
                            results.append(img_data["image"])
                            results.append(img_data["prompt"])
                    else:
                        logger.info("📌 Using text-to-image mode")
                        # Fallback to text-to-image if no images uploaded
                        seed_val = int(seed) if seed != -1 else None
                        generation_results = image_generator.generate_images(
                            prompt=final_prompt,
                            num_images=1,
                            seed=seed_val,
                            enhanced_prompt=base_prompt,  # Original prompt is enhanced
                            edited_prompt=final_prompt,  # Final prompt is edited
                            category="text_to_image",
                        )
                        if not generation_results:
                            raise ValueError("No images were generated")
                        result = generation_results[0]
                        results.append(result["image"])
                        results.append(final_prompt)

                    # Add cooldown between requests
                    if i < len(selected_prompts) - 1:  # Don't wait after last image
                        logger.info(
                            f"⏳ Waiting {COOLDOWN_PERIOD} seconds before next request..."
                        )
                        time.sleep(COOLDOWN_PERIOD)

                except Exception as e:
                    logger.error(f"❌ Failed to generate image {i+1}: {str(e)}")
                    results.append(None)
                    results.append(f"⚠️ Error: {e}")

            # Pad results to always have MAX_IMAGES * 2 items (image, prompt pairs)
            while len(results) < MAX_IMAGES * 2:
                results.append(None)

            # Only return as many outputs as user requested, rest are None
            output = []
            for i in range(MAX_IMAGES):
                if i < num_outputs:
                    output.append(results[i * 2])  # image
                    output.append(results[i * 2 + 1])  # prompt
                else:
                    output.extend([None, None])

            logger.info("✅ Image generation completed")
            # Clear cooldown status
            yield output + [""]

        confirm_btn.click(
            fn=generate_images_from_prompts,
            inputs=[*editable_prompts, num_outputs, seed, prompt, *image_uploads],
            outputs=outputs + [cooldown_status],
        )
    return demo


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 7860))
    demo = create_demo()
    demo.launch(server_port=port, server_name="0.0.0.0", health_check=health_check)
