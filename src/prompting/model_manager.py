import os
import torch
from pathlib import Path
from typing import Optional, Dict
from huggingface_hub import hf_hub_download, HfFolder
from transformers import AutoProcessor, AutoModelForVision2Seq

# === Constants ===
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/app/models")
VISION_MODEL_NAME = os.getenv("VISION_MODEL_NAME", "Salesforce/blip-image-captioning-base")
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "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")

class ModelManager:
    """
    Manages the download and caching of vision models.
    """
    def __init__(
        self,
        model_name: str = VISION_MODEL_NAME,
        cache_dir: str = MODEL_CACHE_DIR,
        hf_token: str = HF_TOKEN
    ):
        """
        Initialize the model manager.

        Args:
            model_name (str): Name of the model to use
            cache_dir (str): Directory to cache models
            hf_token (str): Hugging Face API token
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.hf_token = hf_token
        self._setup_cache()
        self._setup_hf_token()

    def _setup_cache(self):
        """Set up the model cache directory."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"✅ [MODEL] Cache directory ready: {self.cache_dir}")
        except Exception as e:
            print(f"❌ [MODEL ERROR] Failed to create cache directory: {e}")
            raise

    def _setup_hf_token(self):
        """Set up Hugging Face token."""
        try:
            HfFolder.save_token(self.hf_token)
            print("✅ [MODEL] Hugging Face token configured")
        except Exception as e:
            print(f"❌ [MODEL ERROR] Failed to set up HF token: {e}")
            raise

    def download_model(self) -> Dict[str, Path]:
        """
        Download the model and processor files.

        Returns:
            Dict[str, Path]: Paths to downloaded model files
        """
        print(f"\n🎯 [MODEL] Downloading {self.model_name}")
        try:
            # Download model files
            model_path = hf_hub_download(
                repo_id=self.model_name,
                filename="pytorch_model.bin",
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            
            # Download config files
            config_path = hf_hub_download(
                repo_id=self.model_name,
                filename="config.json",
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            
            # Download processor files
            processor_path = hf_hub_download(
                repo_id=self.model_name,
                filename="preprocessor_config.json",
                cache_dir=self.cache_dir,
                token=self.hf_token
            )

            print("✅ [MODEL] Files downloaded successfully")
            return {
                "model": Path(model_path),
                "config": Path(config_path),
                "processor": Path(processor_path)
            }
        except Exception as e:
            print(f"❌ [MODEL ERROR] Failed to download model: {e}")
            raise

    def load_model(self, device: str = "cpu") -> tuple:
        """
        Load the model and processor.

        Args:
            device (str): Device to load the model on

        Returns:
            tuple: (model, processor)
        """
        print(f"\n🎯 [MODEL] Loading model on {device}")
        try:
            # Load processor
            processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token
            )
            
            # Load model
            model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                token=self.hf_token,
                torch_dtype=torch.float32
            ).to(device).eval()

            print("✅ [MODEL] Model loaded successfully")
            return model, processor
        except Exception as e:
            print(f"❌ [MODEL ERROR] Failed to load model: {e}")
            raise

# === Singleton ===
_model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    """
    Get the singleton model manager instance.

    Returns:
        ModelManager: The model manager instance
    """
    return _model_manager 