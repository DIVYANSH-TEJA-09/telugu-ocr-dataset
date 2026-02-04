"""
Split the dataset into 4 sets for parallel processing
Each set contains its own ground_truth and source_images folders
Years split: 1947-1963, 1964-1980, 1981-1996, 1997-2012
"""

import shutil
from pathlib import Path
from tqdm import tqdm

# Config
SOURCE_IMAGES = Path("source_images/images")
GROUND_TRUTH = Path("ground_truth")
OUTPUT_ROOT = Path("OCR-DATASET")

# Year splits (4 sets of ~16-17 years each)
SPLITS = {
    "set1": list(range(1947, 1964)),  # 1947-1963 (17 years)
    "set2": list(range(1964, 1981)),  # 1964-1980 (17 years)
    "set3": list(range(1981, 1997)),  # 1981-1996 (16 years)
    "set4": list(range(1997, 2013)),  # 1997-2012 (16 years)
}

def split_dataset():
    print("🔀 Splitting dataset into 4 sets")
    print("=" * 50)
    
    # Clean and create output
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    
    for set_name, years in SPLITS.items():
        print(f"\n📂 Creating {set_name} (years {years[0]}-{years[-1]})...")
        
        set_dir = OUTPUT_ROOT / set_name
        set_images = set_dir / "source_images" / "images"
        set_gt = set_dir / "ground_truth"
        set_images.mkdir(parents=True)
        set_gt.mkdir(parents=True)
        
        img_count = 0
        gt_count = 0
        
        for year in tqdm(years, desc=f"  {set_name}"):
            year_str = str(year)
            
            # Copy images for this year
            src_year_dir = SOURCE_IMAGES / year_str
            if src_year_dir.exists():
                dst_year_dir = set_images / year_str
                shutil.copytree(src_year_dir, dst_year_dir)
                img_count += sum(1 for _ in dst_year_dir.rglob("*.jpg"))
            
            # Copy GT JSONs for this year
            src_gt_dir = GROUND_TRUTH / year_str
            if src_gt_dir.exists():
                dst_gt_dir = set_gt / year_str
                shutil.copytree(src_gt_dir, dst_gt_dir)
                gt_count += sum(1 for _ in dst_gt_dir.rglob("*.json"))
        
        print(f"   ✅ {set_name}: {img_count} images, {gt_count} GT files")
    
    print("\n" + "=" * 50)
    print("✅ Split Complete!")
    print(f"   Output: {OUTPUT_ROOT.absolute()}")
    print("\nNext: Zip each set and upload to different Google accounts")
    print("   Compress-Archive -Path OCR-DATASET\\set1 -DestinationPath set1.zip")
    print("   Compress-Archive -Path OCR-DATASET\\set2 -DestinationPath set2.zip")
    print("   Compress-Archive -Path OCR-DATASET\\set3 -DestinationPath set3.zip")
    print("   Compress-Archive -Path OCR-DATASET\\set4 -DestinationPath set4.zip")

if __name__ == "__main__":
    split_dataset()
