from milvus_utils import get_record_by_id, connect_to_milvus

def main():
    # Connect to Milvus
    connect_to_milvus()
    
    # Record ID from the logs
    record_id = 458230566761793113
    
    # Get the record
    record = get_record_by_id(record_id)
    
    if record:
        print("\n📝 Record found in Milvus:")
        print(f"Input Prompt: {record['input_prompt']}")
        print(f"Enhanced Prompt: {record['enhanced_prompt']}")
        print(f"Edited Prompt: {record['edited_prompt']}")
        print(f"Model Used: {record['model_used']}")
        print(f"Category: {record['category']}")
        print(f"Timestamp: {record['timestamp']}")
        print(f"Input Image Path: {record['input_image_path']}")
        print(f"Output Image Path: {record['output_image_path']}")
    else:
        print(f"❌ No record found with ID: {record_id}")

if __name__ == "__main__":
    main() 