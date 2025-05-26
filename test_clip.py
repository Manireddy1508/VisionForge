from src.milvus_utils import generate_image_embedding
import numpy as np

def test_clip_embedding():
    # Test image path from user's local file (now in container)
    test_image_path = "/app/car1.webp"
    
    print("\n🔍 Testing CLIP model...")
    try:
        # Generate embedding
        embedding = generate_image_embedding(test_image_path)
        
        # Print embedding details
        print("\n✅ CLIP model is working!")
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding type: {embedding.dtype}")
        print(f"First 5 values: {embedding[:5]}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        
        # Verify embedding properties
        assert embedding.shape == (512,), "Embedding should be 512-dimensional"
        assert not np.isnan(embedding).any(), "Embedding should not contain NaN values"
        assert not np.isinf(embedding).any(), "Embedding should not contain infinite values"
        
        print("\n✅ All embedding properties are correct!")
        
    except Exception as e:
        print(f"\n❌ Error testing CLIP model: {str(e)}")
        raise

if __name__ == "__main__":
    test_clip_embedding() 