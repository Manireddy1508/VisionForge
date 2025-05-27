import os
import time
import base64
import io
import shutil
import tempfile
from typing import List, Optional, Dict, Any

import openai
import requests
from PIL import Image
from dotenv import load_dotenv

from milvus_utils import insert_full_generation_record, _check_for_duplicate

# Load environment variables
load_dotenv()


class ImageGenerator:
    def __init__(self):
        """Initialize the image generator for OpenAI DALL·E 3 (gpt-image-1)."""
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv(
            "OPENAI_IMAGE_MODEL", "gpt-image-1"
        )  # Default to gpt-image-1
        self.temp_dir = os.path.join(os.getcwd(), "temp_images")
        os.makedirs(self.temp_dir, exist_ok=True)
        print(
            f"📌 [DEBUG] Initialized ImageGenerator with OpenAI model: {self.model_name}"
        )
        print(f"📌 [DEBUG] API Key present: {'Yes' if self.api_key else 'No'}")
        print(f"📌 [DEBUG] Temporary directory: {self.temp_dir}")
        if not self.api_key:
            print("⚠️ [WARNING] OPENAI_API_KEY is not set!")

    def _save_image_to_temp(self, image: Image.Image) -> str:
        """Save PIL image to a temporary file and return the path."""
        timestamp = int(time.time())
        temp_path = os.path.join(self.temp_dir, f"gen_{timestamp}.png")
        image.save(temp_path, format="PNG")
        return temp_path

    def cleanup_temp_files(self):
        """Clean up temporary image files older than 24 hours."""
        try:
            current_time = time.time()
            for filename in os.listdir(self.temp_dir):
                filepath = os.path.join(self.temp_dir, filename)
                if os.path.getmtime(filepath) < (current_time - 86400):  # 24 hours
                    os.remove(filepath)
            print(f"✅ [DEBUG] Cleaned up old temporary files")
        except Exception as e:
            print(f"⚠️ [WARNING] Failed to cleanup temporary files: {str(e)}")

    def __del__(self):
        """Cleanup when the object is destroyed."""
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"✅ [DEBUG] Removed temporary directory: {self.temp_dir}")
        except Exception as e:
            print(f"⚠️ [WARNING] Failed to remove temporary directory: {str(e)}")

    def generate_images(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        seed: Optional[int] = None,
        enhanced_prompt: Optional[str] = None,
        edited_prompt: Optional[str] = None,
        category: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate images using OpenAI DALL·E 3 (gpt-image-1).
        """
        print(f"📌 [DEBUG] Generating images with model: {self.model_name}")
        print(f"📌 [DEBUG] Prompt: {prompt}")
        print(f"📌 [DEBUG] Number of images: {num_images}")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set. Please check your .env file.")

        openai.api_key = self.api_key
        results = []
        for i in range(num_images):
            print(f"📌 [DEBUG] Generating image {i+1}/{num_images}")
            try:
                response = openai.images.generate(
                    model=self.model_name, prompt=prompt, n=1, size="1024x1024"
                )
                print(f"📌 [DEBUG] OpenAI response received for image {i+1}")

                if self.model_name == "gpt-image-1":
                    b64_data = response.data[0].b64_json
                    if not b64_data:
                        raise ValueError(
                            "No image data returned by OpenAI. "
                            "Check your prompt and model access."
                        )
                    image_bytes = base64.b64decode(b64_data)
                    pil_img = Image.open(io.BytesIO(image_bytes))
                    image_url = None
                else:
                    image_url = response.data[0].url
                    if not image_url:
                        raise ValueError(
                            "No image URL returned by OpenAI. "
                            "Check your prompt and model access."
                        )
                    image_response = requests.get(image_url)
                    pil_img = Image.open(io.BytesIO(image_response.content))

                # Save image to temp file for Milvus logging
                temp_image_path = self._save_image_to_temp(pil_img)

                # Check for duplicates
                duplicate_id = _check_for_duplicate(prompt, temp_image_path)
                if duplicate_id:
                    print(
                        f"⚠️ [WARNING] Similar image already exists with ID: "
                        f"{duplicate_id}"
                    )

                # Log to Milvus
                try:
                    record_id = insert_full_generation_record(
                        input_prompt=prompt,
                        enhanced_prompt=enhanced_prompt,
                        edited_prompt=edited_prompt,
                        output_image_path=temp_image_path,
                        model_used=self.model_name,
                        category=category,
                    )
                    print(
                        f"✅ [DEBUG] Logged generation to Milvus with ID: {record_id}"
                    )
                except Exception as e:
                    print(f"⚠️ [WARNING] Failed to log to Milvus: {str(e)}")

                results.append(
                    {
                        "image": pil_img,
                        "prompt": prompt,
                        "width": width,
                        "height": height,
                        "signed_url": image_url,
                        "milvus_id": record_id if "record_id" in locals() else None,
                    }
                )
                print(f"✅ [DEBUG] Successfully generated image {i+1}")

            except Exception as e:
                print(f"❌ [ERROR] Failed to generate image {i+1}: {str(e)}")
                raise

        return results

    def edit_images(
        self,
        images: list,  # List of PIL Images
        prompt: str,
        mask_images: list = None,  # Optional: list of PIL Images
        size: str = "1024x1024",
        enhanced_prompt: Optional[str] = None,
        edited_prompt: Optional[str] = None,
        category: Optional[str] = None,
    ) -> list:
        """
        Edit multiple images with a single prompt using gpt-image-1.
        Returns a list of PIL images.
        """
        headers = {"Authorization": f"Bearer {self.api_key}"}
        files = {}
        temp_files = []
        # Save images to temp files for upload
        for idx, img in enumerate(images):
            temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            img.save(temp, format="PNG")
            temp.close()
            files[f"image[]"] = open(temp.name, "rb")
            temp_files.append(temp.name)
        # Optionally add mask images
        if mask_images:
            for idx, mask in enumerate(mask_images):
                temp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                mask.save(temp, format="PNG")
                temp.close()
                files[f"mask"] = open(temp.name, "rb")
                temp_files.append(temp.name)
        data = {"model": "gpt-image-1", "prompt": prompt, "size": size}
        response = requests.post(
            "https://api.openai.com/v1/images/edits",
            headers=headers,
            files=files,
            data=data,
        )
        result = response.json()
        print("OpenAI edit_images response:", result)
        if "data" not in result:
            raise ValueError(f"OpenAI API error: {result.get('error', result)}")
        # Clean up temp files
        for f in temp_files:
            try:
                os.remove(f)
            except Exception:
                pass
        edited_images = []
        for item in result["data"]:
            b64_data = item["b64_json"]
            image_bytes = base64.b64decode(b64_data)
            pil_img = Image.open(io.BytesIO(image_bytes))

            # Save edited image to temp file for Milvus logging
            temp_image_path = self._save_image_to_temp(pil_img)

            # Check for duplicates
            duplicate_id = _check_for_duplicate(prompt, temp_image_path)
            if duplicate_id:
                print(
                    f"⚠️ [WARNING] Similar image already exists with ID: "
                    f"{duplicate_id}"
                )

            # Log to Milvus
            try:
                record_id = insert_full_generation_record(
                    input_prompt=prompt,
                    enhanced_prompt=enhanced_prompt,
                    edited_prompt=edited_prompt,
                    input_image_path=(
                        temp_files[0] if temp_files else None
                    ),  # Use first input image
                    output_image_path=temp_image_path,
                    model_used=self.model_name,
                    category=category,
                )
                print(f"✅ [DEBUG] Logged generation to Milvus with ID: {record_id}")
            except Exception as e:
                print(f"⚠️ [WARNING] Failed to log to Milvus: {str(e)}")

            edited_images.append(
                {
                    "image": pil_img,
                    "prompt": prompt,
                    "milvus_id": record_id if "record_id" in locals() else None,
                }
            )
        return edited_images
