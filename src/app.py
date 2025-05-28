from typing import Dict, Any
import gradio as gr
from src.image_generator import ImageGenerator
from src.prompting.model_manager import ModelManager
from src.prompting.prompt_router import PromptRouter
from src.prompting.prompt_enhancer import PromptEnhancer
from src.prompting.prompt_editor import PromptEditor
from src.prompting.prompt_templates import (
    get_prompt_template,
    get_negative_prompt_template,
)
from src.prompting.prompt_utils import (
    extract_keywords,
    combine_prompts,
    format_prompt,
)
from src.prompting.constants import (
    PROMPT_PREFIX,
    PROMPT_SUFFIX,
    NEGATIVE_PROMPT_PREFIX,
    NEGATIVE_PROMPT_SUFFIX,
)
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get port from environment variable or use default
port = int(os.getenv("PORT", 8080))
logger.info(f"Starting application on port {port}")

# Initialize components
try:
    logger.info("Initializing components...")
    image_generator = ImageGenerator()
    model_manager = ModelManager()
    prompt_router = PromptRouter()
    prompt_enhancer = PromptEnhancer()
    prompt_editor = PromptEditor()
    logger.info("Successfully initialized all components")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

# Load configuration
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    logger.info("Successfully loaded config.json")
except FileNotFoundError:
    logger.warning("config.json not found, using default configuration")
    config = {
        "default_model": "dall-e-3",
        "default_size": "1024x1024",
        "quality": "standard",
        "style": "natural"
    }
except Exception as e:
    logger.error(f"Failed to load config.json: {str(e)}")
    raise

# Initialize UI components
with gr.Blocks(title="AI Image Generator") as demo:
    # Add health check endpoint
    @demo.load()
    def health_check():
        logger.info("Health check endpoint called")
        return "OK"

    gr.Markdown("# AI Image Generator")
    
    with gr.Row():
        with gr.Column():
            # Input components
            prompt = gr.Textbox(
                label="Enter your prompt",
                placeholder="Describe the image you want to generate...",
                lines=3,
            )
            
            negative_prompt = gr.Textbox(
                label="Negative prompt (optional)",
                placeholder="Describe what you don't want in the image...",
                lines=2,
            )
            
            with gr.Row():
                num_images = gr.Slider(
                    minimum=1,
                    maximum=4,
                    value=1,
                    step=1,
                    label="Number of images",
                )
                
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=20.0,
                    value=7.5,
                    step=0.1,
                    label="Guidance scale",
                )
            
            with gr.Row():
                width = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Width",
                )
                
                height = gr.Slider(
                    minimum=256,
                    maximum=1024,
                    value=512,
                    step=64,
                    label="Height",
                )
            
            # Advanced options
            with gr.Accordion("Advanced Options", open=False):
                num_inference_steps = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Inference steps",
                )
                
                seed = gr.Number(
                    value=-1,
                    label="Seed (-1 for random)",
                )
                
                scheduler = gr.Dropdown(
                    choices=["DDIM", "DPMSolverMultistep", "EulerAncestralDiscrete"],
                    value="DPMSolverMultistep",
                    label="Scheduler",
                )
            
            # Generate button
            generate_btn = gr.Button("Generate Images", variant="primary")
        
        with gr.Column():
            # Output components
            gallery = gr.Gallery(
                label="Generated Images",
                show_label=True,
                elem_id="gallery",
                columns=2,
                rows=2,
                object_fit="contain",
            )
            
            # Image details
            with gr.Accordion("Image Details", open=False):
                image_info = gr.JSON(label="Generation Parameters")
    
    # Event handlers
    def generate_images(
        prompt: str,
        negative_prompt: str,
        num_images: int,
        guidance_scale: float,
        width: int,
        height: int,
        num_inference_steps: int,
        seed: int,
        scheduler: str,
    ) -> tuple[list, dict]:
        try:
            # Process prompt
            enhanced_prompt = prompt_enhancer.enhance_prompt(prompt)
            edited_prompt = prompt_editor.edit_prompt(enhanced_prompt)
            final_prompt = prompt_router.route_prompt(edited_prompt)
            
            # Process negative prompt
            if negative_prompt:
                enhanced_neg = prompt_enhancer.enhance_prompt(negative_prompt)
                edited_neg = prompt_editor.edit_prompt(enhanced_neg)
                final_neg = prompt_router.route_prompt(edited_neg)
            else:
                final_neg = ""
            
            # Generate images
            images = []
            for _ in range(num_images):
                image = image_generator.generate_image(
                    prompt=final_prompt,
                    negative_prompt=final_neg,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    num_inference_steps=num_inference_steps,
                    seed=seed,
                    scheduler=scheduler,
                )
                images.append(image)
            
            # Prepare image info
            info = {
                "prompt": final_prompt,
                "negative_prompt": final_neg,
                "parameters": {
                    "guidance_scale": guidance_scale,
                    "width": width,
                    "height": height,
                    "num_inference_steps": num_inference_steps,
                    "seed": seed,
                    "scheduler": scheduler,
                },
            }
            
            return images, info
            
        except Exception as e:
            logger.error(f"Error generating images: {str(e)}")
            raise gr.Error(f"Failed to generate images: {str(e)}")
    
    # Set up event handlers
    generate_btn.click(
        fn=generate_images,
        inputs=[
            prompt,
            negative_prompt,
            num_images,
            guidance_scale,
            width,
            height,
            num_inference_steps,
            seed,
            scheduler,
        ],
        outputs=[gallery, image_info],
    )

# Launch the app
if __name__ == "__main__":
    try:
        logger.info("Starting Gradio interface")
        demo.queue()  # Enable queuing for better performance
        demo.launch(
            server_name="0.0.0.0",
            server_port=port,
            show_error=True,
            debug=True,
            share=False,
            quiet=False
        )
    except Exception as e:
        logger.error(f"Failed to start Gradio interface: {str(e)}")
        raise
