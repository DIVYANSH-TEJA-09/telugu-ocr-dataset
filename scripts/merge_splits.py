"""
Merge 4 set outputs into final dataset
Run after downloading output_set1.zip to output_set4.zip from Colab
"""

import json
import shutil
from pathlib import Path

# Config - Unzip set outputs here
SET_DIRS = [
    Path("output_set1"),
    Path("output_set2"),
    Path("output_set3"),
    Path("output_set4"),
]
OUTPUT_DIR = Path("dataset")

def merge_sets():
    print("🔀 Merging 4 sets into final dataset")
    print("=" * 50)
    
    # Check all sets exist
    missing = [d for d in SET_DIRS if not d.exists()]
    if missing:
        print(f"❌ Missing sets: {missing}")
        print("   Unzip the set outputs first!")
        return
    
    # Setup output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "images").mkdir()
    
    # Merge
    all_metadata = []
    total_images = 0
    
    for set_dir in SET_DIRS:
        print(f"\n📂 Processing {set_dir.name}...")
        
        # Copy images
        img_dir = set_dir / "images"
        if img_dir.exists():
            images = list(img_dir.glob("*.jpg"))
            for img in images:
                shutil.copy2(img, OUTPUT_DIR / "images" / img.name)
            total_images += len(images)
            print(f"   Copied {len(images)} images")
        
        # Load metadata
        meta_file = set_dir / "metadata.jsonl"
        if meta_file.exists():
            count = 0
            with open(meta_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        all_metadata.append(json.loads(line))
                        count += 1
            print(f"   Loaded {count} metadata entries")
    
    # Save merged metadata
    with open(OUTPUT_DIR / "metadata.jsonl", 'w', encoding='utf-8') as f:
        for item in all_metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Stats
    summary = {
        "total_samples": len(all_metadata),
        "total_images": total_images,
        "sets_merged": len(SET_DIRS),
        "min_match_score": 95
    }
    with open(OUTPUT_DIR / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 50)
    print("✅ Merge Complete!")
    print(f"   Total samples: {len(all_metadata)}")
    print(f"   Total images: {total_images}")
    print(f"   Output: {OUTPUT_DIR.absolute()}")
    print("\nNext step: python scripts/upload_to_hf.py")

if __name__ == "__main__":
    merge_sets()
