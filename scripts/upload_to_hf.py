"""
Upload Telugu OCR Dataset to Hugging Face Hub
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login
from datasets import Dataset, Features, Value, Image
import json

# Config
DATASET_DIR = Path("dataset")
HF_REPO = "DIVYANSH-TEJA-09/telugu-ocr-paragraphs"  # Change to your username

def load_dataset():
    """Load the generated dataset from metadata.jsonl"""
    metadata_file = DATASET_DIR / "metadata.jsonl"
    
    if not metadata_file.exists():
        print("❌ Dataset not found. Run build_ocr_dataset.py first!")
        return None
    
    data = []
    with open(metadata_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                img_path = DATASET_DIR / "images" / item["image"]
                if img_path.exists():
                    data.append({
                        "id": item["id"],
                        "image": str(img_path.absolute()),
                        "text": item["text"],
                        "match_score": item["match_score"],
                        "source": item.get("source", ""),
                        "line_count": item.get("line_count", 0)
                    })
    
    print(f"📚 Loaded {len(data)} samples")
    return data

def upload_to_hub():
    print("🚀 Telugu OCR Dataset Upload")
    print("=" * 50)
    
    # Check for HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN not found in environment.")
        print("   Set it with: $env:HF_TOKEN='your_token'")
        print("   Or run: huggingface-cli login")
        return
    
    # Login
    login(token=hf_token)
    
    # Load data
    data = load_dataset()
    if not data:
        return
    
    # Create HF Dataset
    print("📦 Creating Hugging Face Dataset...")
    
    # Convert to format HF expects
    dataset_dict = {
        "id": [d["id"] for d in data],
        "image": [d["image"] for d in data],
        "text": [d["text"] for d in data],
        "match_score": [d["match_score"] for d in data],
        "source": [d["source"] for d in data],
        "line_count": [d["line_count"] for d in data]
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Cast image column to Image type
    dataset = dataset.cast_column("image", Image())
    
    print(f"   Dataset: {dataset}")
    
    # Push to Hub
    print(f"⬆️  Uploading to {HF_REPO}...")
    dataset.push_to_hub(
        HF_REPO,
        private=False,
        commit_message="Upload Telugu OCR paragraph dataset"
    )
    
    print("✅ Upload complete!")
    print(f"   View at: https://huggingface.co/datasets/{HF_REPO}")

if __name__ == "__main__":
    upload_to_hub()
