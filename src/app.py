import dataclasses
import json
from pathlib import Path
import gradio as gr
import torch
import openai
import os
import base64
import io
from uuid import uuid4
from PIL import Image
import time

from src.prompting.prompt_router import PromptRouter
from src.prompting.prompt_enhancer import PromptEnhancer, enhance_prompt_with_similars
from src.prompting.prompt_editor import PromptEditor
from src.prompting.constants import DEFAULT_NUM_PROMPTS
from src.prompting.image_describer import describe_uploaded_images
from src.milvus_utils import (
    connect_to_milvus,
    create_collection,
    generate_image_embedding,
    generate_text_embedding,
    insert_prompt_record,
    insert_image_record,
    search_similar_images,
    update_record_status,
    PROMPT_COLLECTION,
    INPUT_IMAGE_COLLECTION,
    OUTPUT_IMAGE_COLLECTION
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
openai.api_key = os.getenv("OPENAI_API_KEY")

def analyze_image_with_gpt4(image: Image.Image) -> dict:
    """
    Analyze an image using GPT-4 Vision model.
    
    Args:
        image (Image.Image): PIL Image to analyze
        
    Returns:
        dict: Analysis results including description, style, and key elements
    """
    try:
        # Convert PIL Image to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Create the API request
        response = openai.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image in detail. Provide:\n1. A detailed description of what's in the image\n2. The visual style and artistic elements\n3. Key compositional elements\n4. Color palette and lighting\n5. Mood and atmosphere\nFormat the response as a JSON with keys: description, style, composition, colors, mood"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_str}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000
        )
        
        # Parse the response
        content = response.choices[0].message.content
        try:
            analysis = json.loads(content)
        except json.JSONDecodeError:
            # If response isn't valid JSON, create a structured response
            analysis = {
                "description": content,
                "style": "",
                "composition": "",
                "colors": "",
                "mood": ""
            }
        
        return analysis
        
    except Exception as e:
        print(f"❌ [ERROR] Failed to analyze image with GPT-4 Vision: {e}")
        return {
            "description": "",
            "style": "",
            "composition": "",
            "colors": "",
            "mood": ""
        }

def create_demo():
    # Initialize components
    prompt_router = PromptRouter()
    prompt_enhancer = PromptEnhancer()
    prompt_editor = PromptEditor()
    
    # Initialize Milvus connection and ensure collections exist
    try:
        if not connect_to_milvus():
            raise Exception("Failed to connect to Milvus")
            
        print("🔄 Ensuring collections exist...")
        if not create_collection():
            raise Exception("Failed to create collections")
            
        print("✅ Connected to Milvus and collections are ready")
    except Exception as e:
        print(f"⚠️ Failed to initialize Milvus: {str(e)}")
        print("⚠️ Vector search will be disabled.")

    with gr.Blocks() as demo:
        gr.Markdown("## AI Image Generation Playground")
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(label="Prompt", value="handsome woman in the city")
                use_similar_toggle = gr.Checkbox(label="Use similar prompts from previous data?", value=False)
                with gr.Row():
                    image_prompt1 = gr.Image(label="Reference Image", type="pil")
                with gr.Row():
                    with gr.Column():
                        width = gr.Slider(1024, 1536, 1024, step=512, label="Width")
                        height = gr.Slider(1024, 1536, 1024, step=512, label="Height")
                    with gr.Column():
                        gr.Markdown("ℹ️ Tip: Supported sizes are 1024x1024, 1024x1536, and 1536x1024.")
                with gr.Accordion("Advanced Options", open=False):
                    with gr.Row():
                        num_steps = gr.Slider(1, 50, 25, step=1, label="Steps")
                        guidance = gr.Slider(1.0, 5.0, 4.0, step=0.1, label="Guidance Scale")
                        seed = gr.Number(-1, label="Seed (-1 = Random)")
                        num_outputs = gr.Slider(1, 5, DEFAULT_NUM_PROMPTS, step=1, label="Number of Prompts / Images")
                with gr.Row():
                    generate_btn = gr.Button("🎯 Generate Enhanced Prompts")
                    regenerate_btn = gr.Button("🔁 Regenerate")
                    confirm_btn = gr.Button("✅ Confirm & Generate Images")
                # Place editable prompts and outputs here, not in another with gr.Column()
                editable_prompts = []
                outputs = []
                for i in range(DEFAULT_NUM_PROMPTS):
                    editable = gr.Textbox(label=f"Prompt {i+1}", lines=3)
                    image_out = gr.Image(label=f"Image {i+1}")
                    editable_prompts.append(editable)
                    outputs.append(image_out)
                    outputs.append(editable)

        # === Step 1: Analyze Reference Image and Generate Enhanced Prompts ===
        def analyze_and_generate_prompts(prompt, img1, num_outputs, use_similar):
            image_description = ""
            if img1 is not None:
                try:
                    desc = describe_uploaded_images([img1])
                    image_description = desc.get("full_caption") or desc.get("style_description") or ""
                    image_description = prompt_enhancer.clean_repetitive_phrases(image_description)
                    print(f"📝 [DEBUG] Image description: {image_description}")
                except Exception as e:
                    print(f"⚠️ [WARNING] Failed to analyze reference image: {e}")
            
            # Optionally use similar prompts from Milvus if toggle is True
            if use_similar:
                try:
                    from src.milvus_utils import search_by_prompt_similarity
                    similar = search_by_prompt_similarity(prompt, top_k=3)
                    similar_prompts = [r["prompt_record"]["user_prompt"] for r in similar if r.get("prompt_record") and r["prompt_record"].get("user_prompt")]
                    if similar_prompts:
                        print(f"[DEBUG] Using enhance_prompt_with_similars with similar prompts: {similar_prompts}")
                        enhanced_prompts = enhance_prompt_with_similars(prompt, similar_prompts, num_variations=num_outputs)
                    else:
                        print("[DEBUG] No similar prompts found in Milvus. Falling back to user prompt only.")
                        enhanced_prompts = prompt_enhancer.enhance_prompt(
                            prompt=prompt,
                            num_variations=num_outputs,
                            image_description=image_description
                        )
                except Exception as e:
                    print(f"[DEBUG] Milvus search failed: {e}. Falling back to user prompt only.")
                    enhanced_prompts = prompt_enhancer.enhance_prompt(
                        prompt=prompt,
                        num_variations=num_outputs,
                        image_description=image_description
                    )
            else:
                print("[DEBUG] Not using similar prompts from Milvus.")
                print(f"[DEBUG] Enhancing prompt: {prompt}")
                enhanced_prompts = prompt_enhancer.enhance_prompt(
                    prompt=prompt,
                    num_variations=num_outputs,
                    image_description=image_description
                )
                print(f"[DEBUG] Enhanced prompts: {enhanced_prompts}")
            
            # Always return 5 outputs for Gradio, fill unused with ""
            result = [enhanced_prompts[i] if i < len(enhanced_prompts) and i < num_outputs else "" for i in range(5)]
            print(f"[DEBUG] Final prompts to return: {result}")
            return result

        generate_btn.click(
            fn=analyze_and_generate_prompts,
            inputs=[prompt, image_prompt1, num_outputs, use_similar_toggle],
            outputs=editable_prompts
        )

        regenerate_btn.click(
            fn=analyze_and_generate_prompts,
            inputs=[prompt, image_prompt1, num_outputs, use_similar_toggle],
            outputs=editable_prompts
        )
    
        # === Step 2: Confirm and Generate Images ===
        def generate_images_from_prompts(
            prompt1, prompt2, prompt3, prompt4, prompt5,
            width, height, guidance, num_steps, seed,
            img1, num_outputs, base_prompt
        ):
            try:
                # Generate a batch_id for grouping related generations
                batch_id = str(uuid4())
                print(f"🆔 [DEBUG] Generated batch ID: {batch_id}")

                all_prompts = [prompt1, prompt2, prompt3, prompt4, prompt5]
                selected_prompts = [p.strip() for p in all_prompts if p.strip()]

                if not selected_prompts:
                    if not base_prompt or not base_prompt.strip():
                        raise ValueError("⚠️ No prompts available. Please enter a prompt or generate enhanced prompts.")
                    print(f"🧠 [INFO] No enhanced prompts. Using base prompt for {num_outputs} images.")
                    selected_prompts = [base_prompt.strip()] * num_outputs

                if len(selected_prompts) < num_outputs:
                    if len(selected_prompts) == 1:
                        print(f"🧠 [INFO] Only one prompt available. Using it for all {num_outputs} images.")
                        selected_prompts = [selected_prompts[0]] * num_outputs
                    else:
                        raise ValueError(f"⚠️ You selected {num_outputs} images, but only provided {len(selected_prompts)} filled prompts. Please fill or edit prompts.")

                selected_prompts = selected_prompts[:num_outputs]
                
                # Only one reference image allowed
                reference_images = [img1] if img1 is not None else []
                has_reference_images = len(reference_images) > 0
                model_name = "gpt-image-1"
                print(f"\n🤖 [DEBUG] Using model: {model_name} ({'image-to-image' if has_reference_images else 'text-to-image'})")
                
                # Process uploaded image if any
                uploaded_images = []
                image_descriptions = {}
                if has_reference_images:
                    print("\n📸 [DEBUG] Analyzing reference image...")
                    try:
                        image_descriptions = describe_uploaded_images(reference_images)
                        print("\n✅ [DEBUG] Image descriptions:")
                        for key, value in image_descriptions.items():
                            print(f"  - {key}: {value}")
                    except Exception as e:
                        print(f"⚠️ [WARNING] Failed to analyze reference image: {e}")
                        image_descriptions = {}

                    for i, img in enumerate(reference_images):
                        try:
                            if img is not None:
                                # Generate unique session_id for reference image
                                session_id = str(uuid4())
                                input_image_id = str(uuid4())
                                input_dir = Path("input_images")
                                input_dir.mkdir(exist_ok=True, parents=True)
                                input_image_path = input_dir / f"{input_image_id}.png"
                                if img.mode != 'RGB':
                                    img = img.convert('RGB')
                                img.save(input_image_path, format='PNG')
                                input_embedding = generate_image_embedding(str(input_image_path))
                                if input_embedding is not None:
                                    insert_image_record(
                                        session_id=session_id,
                                        input_image_path=str(input_image_path),
                                        output_image_path=None,
                                        input_image_embedding=input_embedding,
                                        output_image_embedding=None,
                                        model_used="Input",
                                        category=None,
                                        status="pending",
                                        version=model_name,
                                        metadata={
                                            "batch_id": batch_id,
                                            "image_index": i + 1,
                                            "original_size": img.size,
                                            "original_mode": img.mode,
                                            "style_description": image_descriptions.get("style_description", ""),
                                            "full_caption": image_descriptions.get("full_caption", ""),
                                            "product_description": image_descriptions.get("product_description", ""),
                                            "intent_description": image_descriptions.get("intent_description", "")
                                        }
                                    )
                                    print(f"✅ [DEBUG] Input image {i+1} saved and embedding stored in Milvus.")
                                uploaded_images.append(str(input_image_path))
                        except Exception as e:
                            print(f"⚠️ [WARNING] Failed to save input image {i+1}: {e}")
                            print(f"Image type: {type(img)}")
                            if isinstance(img, Image.Image):
                                print(f"Image mode: {img.mode}")
                                print(f"Image size: {img.size}")

                print("\n🧠 [DEBUG] Final User-Confirmed Prompts:")
                for i, p in enumerate(selected_prompts):
                    print(f"  [{i+1}] {p}")

                results = []
                for i, final_prompt in enumerate(selected_prompts):
                    try:
                        # Generate unique session_id for each generation
                        session_id = str(uuid4())
                        seed_val = int(seed) if seed != -1 else torch.randint(0, 10**8, (1,)).item()
                        print(f"🧪 [DEBUG] Using seed: {seed_val} for image {i+1}")
                        prompt_embedding = generate_text_embedding(final_prompt)
                        if prompt_embedding is None:
                            print(f"⚠️ [WARNING] Failed to generate text embedding for prompt {i+1}")
                            prompt_embedding = []
                        
                        # Use the base_prompt parameter which contains the user's actual input
                        user_prompt_value = base_prompt
                        print(f"📝 [DEBUG] User prompt value: {user_prompt_value}")
                        
                        insert_prompt_record(
                            session_id=session_id,
                            user_prompt=user_prompt_value,
                            enhanced_prompt=final_prompt,
                            final_prompt=final_prompt,
                            prompt_embedding=prompt_embedding,
                            status="pending",
                            version=model_name,
                            metadata={
                                "batch_id": batch_id,
                                "generation_index": i + 1,
                                "seed": seed_val,
                                "width": width,
                                "height": height,
                                "guidance": guidance,
                                "num_steps": num_steps,
                                "has_reference_images": has_reference_images
                            }
                        )
                        # Always use gpt-image-1 for generation
                        if has_reference_images:
                            try:
                                # Save the reference image to a temporary file
                                temp_image_path = "temp_reference_image.png"
                                reference_images[0].save(temp_image_path, format="PNG")
                                with open(temp_image_path, "rb") as img_file:
                                    response = openai.images.edit(
                                        model="gpt-image-1",
                                        image=img_file,
                                        prompt=final_prompt,
                                        n=1,
                                        size=f"{int(width)}x{int(height)}"
                                    )
                                if not response or not hasattr(response, 'data') or not response.data:
                                    raise ValueError("No image data in response")
                                image_data = base64.b64decode(response.data[0].b64_json)
                            except Exception as e:
                                print(f"⚠️ [WARNING] Image edit failed, falling back to text-to-image: {e}")
                                try:
                                    response = openai.images.generate(
                                        model="gpt-image-1",
                                        prompt=final_prompt,
                                        n=1,
                                        size=f"{int(width)}x{int(height)}",
                                        quality="high"
                                    )
                                    if not response or not hasattr(response, 'data') or not response.data:
                                        raise ValueError("No image data in text-to-image response")
                                    image_data = base64.b64decode(response.data[0].b64_json)
                                except openai.BadRequestError as e:
                                    if "moderation_blocked" in str(e):
                                        raise ValueError("⚠️ This prompt was rejected by the safety system. Please avoid using copyrighted characters, brands, or content that may violate OpenAI's content policy.")
                                    raise e
                        else:
                            try:
                                response = openai.images.generate(
                                    model="gpt-image-1",
                                    prompt=final_prompt,
                                    n=1,
                                    size=f"{int(width)}x{int(height)}",
                                    quality="high"
                                )
                                if not response or not hasattr(response, 'data') or not response.data:
                                    raise ValueError("No image data in text-to-image response")
                                image_data = base64.b64decode(response.data[0].b64_json)
                            except openai.BadRequestError as e:
                                if "moderation_blocked" in str(e):
                                    raise ValueError("⚠️ This prompt was rejected by the safety system. Please avoid using copyrighted characters, brands, or content that may violate OpenAI's content policy.")
                                raise e
                        gen_image = Image.open(io.BytesIO(image_data))
                        print(f"✅ [DEBUG] Image {i+1} generated successfully.")
                        try:
                            image_id = str(uuid4())
                            output_dir = Path("output")
                            output_dir.mkdir(exist_ok=True)
                            image_path = output_dir / f"{image_id}.png"
                            gen_image.save(image_path)
                            image_embedding = generate_image_embedding(str(image_path))
                            if image_embedding is not None:
                                update_record_status(
                                    PROMPT_COLLECTION,
                                    session_id,
                                    "complete",
                                    {"generation_completed_at": int(time.time())}
                                )
                                insert_image_record(
                                    session_id=session_id,
                                    input_image_path=uploaded_images[0] if uploaded_images else None,
                                    output_image_path=str(image_path),
                                    input_image_embedding=None,
                                    output_image_embedding=image_embedding,
                                    model_used=model_name,
                                    category=None,
                                    status="complete",
                                    version=model_name,
                                    metadata={
                                        "batch_id": batch_id,
                                        "generation_index": i + 1,
                                        "seed": seed_val,
                                        "width": width,
                                        "height": height,
                                        "guidance": guidance,
                                        "num_steps": num_steps,
                                        "generation_completed_at": int(time.time()),
                                        "has_reference_images": has_reference_images
                                    }
                                )
                                print(f"✅ [DEBUG] Image {i+1} embedding stored in Milvus.")
                        except Exception as e:
                            print(f"⚠️ [WARNING] Failed to store image {i+1} embedding: {e}")
                            print(f"Image path: {image_path}")
                            print(f"Image type: {type(gen_image)}")
                            if isinstance(gen_image, Image.Image):
                                print(f"Image mode: {gen_image.mode}")
                                print(f"Image size: {gen_image.size}")
                        results.append(gen_image)
                    except Exception as e:
                        print(f"⚠️ [WARNING] Failed to generate image {i+1}: {e}")
                        print(f"Prompt: {final_prompt}")
                        print(f"Seed: {seed_val}")
                        print(f"Width: {width}")
                        print(f"Height: {height}")
                        print(f"Guidance: {guidance}")
                        print(f"Num steps: {num_steps}")
                        print(f"Has reference_images: {has_reference_images}")
                        print(f"Model: {model_name}")
                        print(f"Error type: {type(e)}")
                        print(f"Error message: {str(e)}")
                        if hasattr(e, 'response'):
                            print(f"Response status: {e.response.status_code}")
                            print(f"Response text: {e.response.text}")
                        raise e

                formatted_results = []
                for i in range(int(num_outputs)):
                    if i < len(results):
                        formatted_results.extend([results[i], selected_prompts[i]])
                    else:
                        formatted_results.extend([None, ""])
                while len(formatted_results) < 2 * DEFAULT_NUM_PROMPTS:
                    formatted_results.extend([None, ""])
                return formatted_results
            except Exception as e:
                print(f"❌ [ERROR] Generation failed: {e}")
                raise e

        confirm_btn.click(
            fn=generate_images_from_prompts,
            inputs=[
                *editable_prompts, width, height, guidance, num_steps,
                seed, image_prompt1, num_outputs, prompt
            ],
            outputs=outputs
        )

        # === Step 3: Similar Image Search ===
        with gr.Accordion("🔍 Similar Image Search", open=False):
            with gr.Row():
                search_image = gr.Image(label="Search Image", type="pil")
                search_btn = gr.Button("🔍 Find Similar Images")
            
            with gr.Row():
                similar_images = gr.Gallery(label="Similar Images", show_label=True, columns=3, rows=2, height=400)
                similar_prompts = gr.JSON(label="Similar Image Details")

            def find_similar_images(image):
                if image is None:
                    return None, "Please upload an image to search."
                
                try:
                    # Generate embedding for search image
                    embedding = generate_image_embedding(image)
                    if not embedding:
                        return None, "Failed to generate image embedding."
                    
                    # Search for similar images
                    similar = search_similar_images(embedding, top_k=6)
                    if not similar:
                        return None, "No similar images found."
                    
                    # Load and return similar images
                    images = []
                    details = []
                    for item in similar:
                        try:
                            img = Image.open(item["image_path"])
                            images.append(img)
                            details.append({
                                "prompt": item["prompt"],
                                "distance": float(item["distance"])
                            })
                        except Exception as e:
                            print(f"⚠️ [WARNING] Failed to load similar image: {e}")
                    return images, details
                except Exception as e:
                    return None, f"Search failed: {str(e)}"

            search_btn.click(
                fn=find_similar_images,
                inputs=[search_image],
                outputs=[similar_images, similar_prompts]
            )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_port=7860)
