from src.milvus_utils import get_record_by_id, search_similar_images, connect_to_milvus

def main():
    # Connect to Milvus
    connect_to_milvus()
    
    # IDs of the two images
    image1_id = 458230566761793113
    image2_id = 458230566761793115
    
    # Get both records
    record1 = get_record_by_id(image1_id)
    record2 = get_record_by_id(image2_id)
    
    print("\n📝 First Image Details:")
    print(f"ID: {image1_id}")
    print(f"Input Prompt: {record1['input_prompt']}")
    print(f"Enhanced Prompt: {record1['enhanced_prompt']}")
    print(f"Output Image Path: {record1['output_image_path']}")
    
    print("\n📝 Second Image Details:")
    print(f"ID: {image2_id}")
    print(f"Input Prompt: {record2['input_prompt']}")
    print(f"Enhanced Prompt: {record2['enhanced_prompt']}")
    print(f"Output Image Path: {record2['output_image_path']}")
    
    # Search for similar images using the second image
    print("\n🔍 Searching for similar images using the second image...")
    similar_images = search_similar_images(
        image_path=record2['output_image_path'],
        top_k=3,
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
            print(f"Timestamp: {img['timestamp']}")
    else:
        print("\n❌ No similar images found")

if __name__ == "__main__":
    main() 