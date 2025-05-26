from src.image_generator import ImageGenerator
from src.milvus_utils import search_similar_images, connect_to_milvus
import time

def main():
    # Initialize image generator
    generator = ImageGenerator()
    
    # Original prompt
    original_prompt = "a beautiful sunset over mountains"
    
    # Modified prompt
    modified_prompt = "a stunning sunset over majestic mountains with golden rays"
    
    print("\n🎨 Generating new image with modified prompt...")
    results = generator.generate_images(
        prompt=modified_prompt,
        category="text_to_image"
    )
    
    if results and results[0].get("image"):
        # Get the path of the generated image
        image_path = results[0].get("output_image_path")
        print(f"\n✅ Generated new image at: {image_path}")
        
        # Connect to Milvus
        connect_to_milvus()
        
        # Search for similar images
        print("\n🔍 Searching for similar images in Milvus...")
        similar_images = search_similar_images(
            image_path=image_path,
            top_k=3,  # Get top 3 similar images
            category="text_to_image"
        )
        
        if similar_images:
            print("\n📝 Found similar images:")
            for idx, img in enumerate(similar_images, 1):
                print(f"\n--- Similar Image {idx} ---")
                print(f"ID: {img['id']}")
                print(f"Similarity Distance: {img['distance']:.4f}")
                print(f"Input Prompt: {img['input_prompt']}")
                print(f"Enhanced Prompt: {img['enhanced_prompt']}")
                print(f"Model Used: {img['model_used']}")
                print(f"Timestamp: {img['timestamp']}")
        else:
            print("\n❌ No similar images found")
    else:
        print("\n❌ Failed to generate image")

if __name__ == "__main__":
    main() 