"""
Telugu OCR Dataset Builder
Processes all ~35k page images, extracts paragraphs via Tesseract,
fuzzy-matches with ground truth, keeps only high-confidence (≥95%) pairs.
"""

import pytesseract
import cv2
import numpy as np
import json
import shutil
import re
from pathlib import Path
from collections import defaultdict
from rapidfuzz import fuzz
from tqdm import tqdm

# Config
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

SOURCE_IMAGES_DIR = Path("source_images/images")  # Year/PDF_NAME/page_XXX.jpg
GROUND_TRUTH_DIR = Path("ground_truth")           # Year/json_file.json
OUTPUT_DIR = Path("dataset")
MIN_LINES = 4
MIN_MATCH_SCORE = 95

def load_image_unicode(path):
    stream = open(path, "rb")
    bytes_data = bytearray(stream.read())
    stream.close()
    numpyarray = np.asarray(bytes_data, dtype=np.uint8)
    return cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)

def save_image_unicode(img, path):
    is_success, buffer = cv2.imencode(".jpg", img)
    if is_success:
        buffer.tofile(str(path))

def split_into_sentences(text):
    chunks = re.split(r'[.\n।]+', text)
    sentences = [s.strip() for s in chunks if s.strip() and len(s.strip()) > 5]
    return sentences

def find_best_sentence_range(tesseract_text, sentences):
    if not tesseract_text.strip() or not sentences:
        return None, 0
    
    clean_tess = " ".join(tesseract_text.split())
    tess_len = len(clean_tess)
    
    best_match = None
    best_score = 0
    n = len(sentences)
    
    for size in range(1, min(16, n + 1)):
        for start in range(n - size + 1):
            combined = " ".join(sentences[start:start + size])
            if len(combined) < tess_len * 0.7:
                continue
            score = fuzz.ratio(clean_tess, combined)
            if score > best_score:
                best_score = score
                best_match = combined
    
    return best_match, best_score

def build_story_index():
    """
    Build index: (pdf_stem, page_num) -> story_content
    Each JSON contains multiple stories with pdf_page_start and pdf_page_end
    """
    print("📚 Building story index from JSON files...")
    
    # Map: (pdf_stem, page_num) -> content
    page_to_story = {}
    
    for json_file in tqdm(list(GROUND_TRUTH_DIR.rglob("*.json")), desc="Indexing JSONs"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except:
            continue
        
        # JSON stem: "చందమామ_1947_07"
        pdf_stem = json_file.stem
        
        # Each JSON has a "stories" list or might be a single story
        stories = data.get("stories", [])
        if not stories and "story" in data:
            stories = [data["story"]]
        if not stories and "content" in data:
            stories = [data]
        
        for story in stories:
            content = story.get("content", "")
            if not content:
                continue
            
            start_page = story.get("pdf_page_start", 1)
            end_page = story.get("pdf_page_end", start_page)
            
            # Map each page in range to this story's content
            for page in range(start_page, end_page + 1):
                key = (pdf_stem, page)
                # If multiple stories on same page, concatenate
                if key in page_to_story:
                    page_to_story[key] += "\n" + content
                else:
                    page_to_story[key] = content
    
    print(f"   Indexed {len(page_to_story)} page-story mappings")
    return page_to_story

def parse_page_number(filename):
    """Extract page number from filename like 'page_007.jpg'"""
    stem = Path(filename).stem
    # Try various patterns
    match = re.search(r'page[_-]?(\d+)', stem, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Try just trailing number
    match = re.search(r'(\d+)$', stem)
    if match:
        return int(match.group(1))
    return None

def extract_and_align(img_path, gt_text, output_images_dir, unique_id):
    """Extract paragraphs from image, align with GT, return high-confidence matches"""
    results = []
    
    img = load_image_unicode(img_path)
    if img is None:
        return results
    
    try:
        data = pytesseract.image_to_data(img, lang='tel', output_type=pytesseract.Output.DICT)
    except:
        return results
    
    # Group words by paragraph
    paragraphs = defaultdict(list)
    n_boxes = len(data['text'])
    
    for i in range(n_boxes):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        if conf > 0 and text:
            key = (data['block_num'][i], data['par_num'][i])
            paragraphs[key].append({
                'text': text,
                'x': data['left'][i],
                'y': data['top'][i],
                'w': data['width'][i],
                'h': data['height'][i],
                'line_num': data['line_num'][i]
            })
    
    # Split GT into sentences
    sentences = split_into_sentences(gt_text)
    if not sentences:
        return results
    
    img_h, img_w = img.shape[:2]
    para_idx = 0
    
    for key, words in paragraphs.items():
        if not words:
            continue
        
        # Compute bounding box
        x_min = min(w['x'] for w in words)
        y_min = min(w['y'] for w in words)
        x_max = max(w['x'] + w['w'] for w in words)
        y_max = max(w['y'] + w['h'] for w in words)
        
        pad = 10
        x_min = max(0, x_min - pad)
        y_min = max(0, y_min - pad)
        x_max = min(img_w, x_max + pad)
        y_max = min(img_h, y_max + pad)
        
        width = x_max - x_min
        height = y_max - y_min
        if width < 50 or height < 20:
            continue
        
        # Group by line
        sorted_words = sorted(words, key=lambda w: (w['line_num'], w['x']))
        lines_dict = defaultdict(list)
        for w in sorted_words:
            lines_dict[w['line_num']].append(w['text'])
        
        # Check minimum lines
        if len(lines_dict) < MIN_LINES:
            continue
        
        # Get Tesseract text
        tess_text = '\n'.join(' '.join(ws) for ws in lines_dict.values())
        
        # Fuzzy match
        matched_gt, score = find_best_sentence_range(tess_text, sentences)
        
        # Only keep high confidence
        if score < MIN_MATCH_SCORE or not matched_gt:
            continue
        
        # Crop and save
        para_crop = img[y_min:y_max, x_min:x_max]
        para_id = f"{unique_id}_para_{para_idx:02d}"
        crop_path = output_images_dir / f"{para_id}.jpg"
        save_image_unicode(para_crop, crop_path)
        
        results.append({
            "id": para_id,
            "image": f"{para_id}.jpg",
            "text": matched_gt,
            "match_score": score,
            "source": str(img_path.relative_to(SOURCE_IMAGES_DIR)),
            "line_count": len(lines_dict)
        })
        
        para_idx += 1
    
    return results

def run_full_pipeline():
    print("🚀 Full-Scale Telugu OCR Dataset Pipeline v2")
    print("   Processing ALL images from full_dataset/images/")
    print("=" * 60)
    
    # Setup output
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True)
    (OUTPUT_DIR / "images").mkdir()
    
    # Build story index
    page_to_story = build_story_index()
    
    # Get all images
    all_images = list(SOURCE_IMAGES_DIR.rglob("*.jpg"))
    print(f"📷 Total images to process: {len(all_images)}")
    
    # Process
    all_results = []
    stats = {"processed": 0, "with_gt": 0, "extracted": 0}
    
    for img_path in tqdm(all_images, desc="Processing"):
        stats["processed"] += 1
        
        # Parse: full_dataset/images/1947/చందమామ 1947 07/page_007.jpg
        # PDF folder name: "చందమామ 1947 07"
        # JSON stem: "చందమామ_1947_07" (spaces -> underscores)
        pdf_folder = img_path.parent.name
        pdf_stem = pdf_folder.replace(" ", "_")
        page_num = parse_page_number(img_path.name)
        
        if page_num is None:
            continue
        
        # Look up GT
        key = (pdf_stem, page_num)
        gt_text = page_to_story.get(key)
        
        if not gt_text:
            continue
        
        stats["with_gt"] += 1
        
        # Create unique ID for this page
        year = img_path.parent.parent.name
        unique_id = f"{year}_{pdf_stem}_p{page_num:03d}"
        
        results = extract_and_align(img_path, gt_text, OUTPUT_DIR / "images", unique_id)
        
        if results:
            stats["extracted"] += len(results)
            all_results.extend(results)
    
    # Save metadata
    with open(OUTPUT_DIR / "metadata.jsonl", 'w', encoding='utf-8') as f:
        for item in all_results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    # Save summary
    summary = {
        "total_images_processed": stats["processed"],
        "images_with_gt": stats["with_gt"],
        "high_confidence_paragraphs": stats["extracted"],
        "min_match_score": MIN_MATCH_SCORE,
        "min_lines": MIN_LINES
    }
    with open(OUTPUT_DIR / "summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print(f"   Images processed: {stats['processed']}")
    print(f"   Images with GT: {stats['with_gt']}")
    print(f"   High-confidence paragraphs (≥{MIN_MATCH_SCORE}%): {stats['extracted']}")
    print(f"   Output: {OUTPUT_DIR.absolute()}")

if __name__ == "__main__":
    run_full_pipeline()
