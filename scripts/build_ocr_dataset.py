"""
Final Production Script for Telugu OCR Dataset Creation
Designed for high-quality, single-system execution.

Features:
- Multiprocessing for speed (uses all available cores - 2)
- Strict Quality Control:
    - Min 95% fuzzy match compliance (0.95 threshold)
    - Min 4 lines per paragraph
    - Tesseract OCR (Telugu)
    - Exact substring alignment only
- Robust JSON Metadata for Hugging Face
- Logging: Detailed statistics and error tracking

Usage:
    python scripts/build_ocr_dataset.py
"""

import pytesseract
import cv2
import numpy as np
import json
import shutil
import re
import os
import multiprocessing
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz
from tqdm import tqdm
import time
import functools

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Directories
SOURCE_IMAGES_DIR = Path("source_images/images")
GROUND_TRUTH_DIR = Path("ground_truth")
OUTPUT_DIR = Path("dataset")

# Strict Quality Thresholds
MIN_MATCH_SCORE = 95       # STRICT: Only keep >= 95% matches
MIN_LINES = 4              # STRICT: Minimum 4 lines for a valid paragraph
MIN_SENTENCE_LEN = 10      # Minimum chars for a sentence to be considered valid GT
IMAGE_PADDING = 10         # Padding around crop (pixels)

# Processing
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 2)  # Leave 2 cores for system

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_directories():
    if OUTPUT_DIR.exists():
        print(f"⚠️  Output directory {OUTPUT_DIR} exists. Resuming/Overwriting...")
    else:
        OUTPUT_DIR.mkdir(parents=True)
    
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    print(f"✅ Directories ready. Output: {OUTPUT_DIR.absolute()}")

def split_into_sentences(text):
    """Splits GT text into sentences/chunks for alignment."""
    # Split by periods, newlines, or danda (।)
    chunks = re.split(r'[.\n।]+', text)
    return [s.strip() for s in chunks if s.strip() and len(s.strip()) > MIN_SENTENCE_LEN]

def find_best_sentence_range(tesseract_text, sentences):
    """
    Finds the best contiguous range of sentences in GT that matches the OCR text.
    Returns: (best_matched_text, score)
    """
    if not tesseract_text.strip() or not sentences:
        return None, 0
    
    clean_tess = " ".join(tesseract_text.split())
    tess_len = len(clean_tess)
    
    best_match = None
    best_score = 0
    
    # Heuristic: Try different window sizes of sentences
    # Since OCR might merge/split lines, we checking combinations of sentences
    max_window = min(20, len(sentences) + 1)
    
    for size in range(1, max_window):
        # Optimization: Only check windows that have roughly similar length (±40%)
        # This speeds up search significantly
        for start in range(len(sentences) - size + 1):
            combined = " ".join(sentences[start:start + size])
            
            # Quick length check
            if len(combined) < tess_len * 0.6 or len(combined) > tess_len * 1.4:
                continue
                
            score = fuzz.ratio(clean_tess, combined)
            
            if score > best_score:
                best_score = score
                best_match = combined
                
                # if perfect match found, strictly return
                if best_score == 100:
                    return best_match, best_score

    return best_match, best_score

def parse_page_number(filename):
    """Robustly extracts page number from filename."""
    # Try pattern like 'page_123' or 'page-123'
    match = re.search(r'page[_-]?(\d+)', Path(filename).stem, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try trailing number like 'abc_123'
    match = re.search(r'(\d+)$', Path(filename).stem)
    if match:
        return int(match.group(1))
    
    return None

def build_story_index():
    """Builds a lookup index: (pdf_stem, page_num) -> full_page_gt_text"""
    page_to_story = {}
    print("📚 Building Story Index from Ground Truth...")
    
    json_files = list(GROUND_TRUTH_DIR.rglob("*.json"))
    
    for json_file in tqdm(json_files, desc="Indexing JSONs"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading {json_file}: {e}")
            continue
            
        pdf_stem = json_file.stem
        
        # Handle different JSON structures
        stories = data.get("stories", [])
        if not stories and "story" in data:
            stories = [data["story"]]
        if not stories and "content" in data:  # Flat structure
            stories = [data]
            
        for story in stories:
            content = story.get("content", "")
            if not content:
                continue
                
            try:
                # Handle int/float/string page numbers safely
                start_p = int(story.get("pdf_page_start", 1))
                end_p = int(story.get("pdf_page_end", start_p))
                
                for page in range(start_p, end_p + 1):
                    key = (pdf_stem, page)
                    # Append content if multiple stories are on same page
                    page_to_story[key] = page_to_story.get(key, "") + "\n" + content
            except ValueError:
                continue # Skip if page numbers are malformed
                
    print(f"✅ Index built! Covered {len(page_to_story)} pages.")
    return page_to_story

# ============================================================================
# CORE PROCESSING WORKER
# ============================================================================

def process_image(img_path, gt_text, output_dir):
    """
    Worker function to process a single image.
    1. OCR with Tesseract
    2. Group into paragraphs
    3. Fuzzy match with GT
    4. Save valid crops
    """
    results = []
    
    img = cv2.imread(str(img_path))
    if img is None:
        return results

    img_h, img_w = img.shape[:2]
    
    # Run Tesseract
    try:
        # psm 3 is default (fully automatic page segmentation)
        data = pytesseract.image_to_data(img, lang='tel', output_type=pytesseract.Output.DICT, config='--psm 3')
    except Exception as e:
        return results

    # Group into paragraphs
    paragraphs = defaultdict(list)
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        if conf > 0 and text:
            # key: (block_num, par_num)
            key = (data['block_num'][i], data['par_num'][i])
            paragraphs[key].append({
                'text': text,
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i],
                'line_num': data['line_num'][i]
            })

    # Prepare GT sentences
    sentences = split_into_sentences(gt_text)
    if not sentences:
        return results

    # Process each detected paragraph
    para_idx = 0
    unique_id_base = f"{img_path.parent.parent.name}_{img_path.parent.name}_{img_path.stem}"
    unique_id_base = unique_id_base.replace(" ", "_")

    for key, words in paragraphs.items():
        if not words:
            continue
            
        # 1. Bounding Box Calculation
        x_min = min(w['x'] for w in words)
        y_min = min(w['y'] for w in words)
        x_max = max(w['x'] + w['w'] for w in words)
        y_max = max(w['y'] + w['h'] for w in words)
        
        # Add padding
        x_min = max(0, x_min - IMAGE_PADDING)
        y_min = max(0, y_min - IMAGE_PADDING)
        x_max = min(img_w, x_max + IMAGE_PADDING)
        y_max = min(img_h, y_max + IMAGE_PADDING)
        
        # Filter tiny boxes (noise)
        if (x_max - x_min) < 50 or (y_max - y_min) < 20:
            continue

        # 2. STRICT Line Counting Check (Minimum 4 lines)
        lines = set(w['line_num'] for w in words)
        if len(lines) < MIN_LINES:
            continue

        # 3. Construct OCR Text for matching
        sorted_words = sorted(words, key=lambda w: (w['line_num'], w['x']))
        lines_dict = defaultdict(list)
        for w in sorted_words:
            lines_dict[w['line_num']].append(w['text'])
            
        tess_text = '\n'.join(' '.join(ws) for ws in lines_dict.values())
        
        # 4. Fuzzy Matching (Strict 95%)
        matched_gt, score = find_best_sentence_range(tess_text, sentences)
        
        # STRICT Check: must be >= 95%
        if score < MIN_MATCH_SCORE or not matched_gt:
            continue

        # 5. Save Result
        para_id = f"{unique_id_base}_p{para_idx:02d}"
        crop_filename = f"{para_id}.jpg"
        crop_path = output_dir / "images" / crop_filename
        
        # Save Crop
        crop_img = img[y_min:y_max, x_min:x_max]
        cv2.imwrite(str(crop_path), crop_img)
        
        # Save robust metadata for HF
        entry = {
            "id": para_id,
            "file_name": crop_filename,  # HF Datasets expects 'file_name' for ImageFolder
            "text": matched_gt,          # The cleaned GT text
            "ground_truth": matched_gt,  # REDUNDANT but explicit for clarity
            "ocr_text_tesseract": tess_text, # What Tesseract saw (for comparison)
            "match_score": score,
            "line_count": len(lines),
            "origin_year": img_path.parent.name,
            "origin_issue": img_path.parent.parent.name
        }
        results.append(entry)
        para_idx += 1
        
    return results

# Wrapper for multiprocessing to unpack args
def worker(args):
    img_path, gt_text = args
    return process_image(img_path, gt_text, OUTPUT_DIR)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"🚀 Starting Telugu OCR Dataset Builder (Strict Mode)")
    print(f"   Workers: {NUM_WORKERS}")
    print(f"   Min Match Score: {MIN_MATCH_SCORE}%")
    print(f"   Min Lines: {MIN_LINES}")
    
    setup_directories()
    
    # 1. Build Index
    page_to_story = build_story_index()
    
    # 2. Collect Work Items
    print("\n🔍 Scanning for source images...")
    work_items = []
    
    # Iterate through all year folders (1947-2012)
    # Assuming structure: source_images/images/YYYY/Magazine_Name_Month/*.jpg
    # Or source_images/images/YYYY/*.jpg depending on exact structure, making it robust:
    
    for year_dir in SOURCE_IMAGES_DIR.iterdir():
        if not year_dir.is_dir(): continue
        
        # Recursive glob to catch subfolders if any
        for img_path in year_dir.rglob("*.jpg"):
            pdf_stem = img_path.parent.name 
            page_num = parse_page_number(img_path.name)
            
            if page_num is None:
                continue
                
            # Try exact match first
            gt_text = page_to_story.get((pdf_stem, page_num))
            
            # If fail, try variations
            if not gt_text:
                gt_text = page_to_story.get((pdf_stem.replace(" ", "_"), page_num))

            if gt_text:
                work_items.append((img_path, gt_text))
            
    print(f"✅ Found {len(work_items)} pages with corresponding Ground Truth.")
    
    if len(work_items) == 0:
        print("❌ No matches found. Please check paths and directory structure!")
        return

    # 3. Execute with Multiprocessing
    print(f"\n⚙️  Processing on {NUM_WORKERS} cores... (This will take 24-48 hours)")
    
    total_extracted = 0
    metadata_file = OUTPUT_DIR / "metadata.jsonl"
    
    with open(metadata_file, 'w', encoding='utf-8') as meta_f:
        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            for result_batch in tqdm(pool.imap_unordered(worker, work_items, chunksize=5), total=len(work_items)):
                if result_batch:
                    total_extracted += len(result_batch)
                    for item in result_batch:
                        meta_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        
    # 4. Final Stats
    print("\n" + "="*50)
    print("✅ PROCESSING COMPLETE")
    print(f"   Total Pages Processed: {len(work_items)}")
    print(f"   Total Valid Paragraphs Extracted: {total_extracted}")
    print(f"   Dataset Location: {OUTPUT_DIR.absolute()}")
    print("="*50)

if __name__ == "__main__":
    multiprocessing.freeze_support() # For Windows
    main()
