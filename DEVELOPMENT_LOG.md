# Telugu OCR Dataset Pipeline - Development Log

## Project Overview
Building a high-quality Telugu OCR training dataset from ~35,000 Chandamama magazine page images with paragraph-level ground truth text.

---

## Approach History

### 1. Strict CV-Based Pipeline (FAILED)
**Goal:** Use computer vision to detect paragraphs, match count with GT, crop and align.

**Implementation:**
- Binarization → Column detection → Line detection → Paragraph grouping
- Hard rejection rule: `detected_paragraphs != GT_paragraphs` → reject

**Results:**
- Initial: 0.5% acceptance rate
- After Unicode fix: ~8% acceptance rate

**Failures:**
1. **Unicode Filename Error:** `cv2.imwrite()` failed on Telugu filenames. Fixed with `cv2.imencode()` + `buffer.tofile()`.
2. **Count Mismatch Rejections:** 90%+ pages rejected due to paragraph count mismatch.
3. **Rescue Script Failed:** Attempted to tune gap thresholds for mismatched pages - introduced corrupted mappings.

**Verdict:** ❌ Too brittle. CV heuristics don't generalize across magazine layouts.

---

### 2. Single-Page Story Filtering (PARTIAL SUCCESS)
**Goal:** Filter dataset to only single-page stories (where pdf_page_start == pdf_page_end) for simpler alignment.

**Implementation:**
- `filter_single_pages.py` → Extracted 2,618 single-page stories
- Each story = 1 image + 1 complete GT text

**Results:**
- 2,618 clean single-page stories extracted
- Verification UI built for manual review

**Verdict:** ✅ Good intermediate step, but limited to single-page content.

---

### 3. Tesseract OCR Text Detection (EXPERIMENT)
**Goal:** Use Tesseract to detect text regions instead of custom CV.

**Experiments:**
1. **Word-level:** Too granular, noisy
2. **Line-level:** 108 lines from 5 images - good for CRNN training
3. **Paragraph-level:** 30 paragraphs from 5 images - better for alignment

**Verdict:** ✅ Tesseract paragraph detection is robust. Proceed with this.

---

### 4. Fuzzy Text Alignment (FINAL APPROACH)
**Goal:** Match Tesseract OCR output to GT text using fuzzy matching.

**Evolution:**
1. **v1 - Full Paragraph Match:** Matched Tesseract text to entire GT paragraphs. Problem: Same GT shown for multiple crops.
2. **v2 - Sliding Window Substring:** Extracted exact substring from GT. Problem: Boundaries cut mid-sentence.
3. **v3 - Sentence-Based Matching:** Split GT by periods/newlines, match consecutive sentences. ✅ Clean boundaries.

**Final Parameters:**
- Min lines per paragraph: 4 (filters noise like headers/footers)
- Min match score: 95% (high confidence only)
- Max sentence range: 15 consecutive sentences

**Results (5 years sample):**
- 451 aligned paragraphs
- 44.8% achieved ≥95% match score
- Mean score: 90.8%

**Verdict:** ✅ This is the production approach.

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Paragraph detection | Tesseract | More robust than custom CV |
| Text alignment | Fuzzy (rapidfuzz) | Handles OCR errors gracefully |
| Match threshold | 95% | High confidence for training |
| Min lines | 4 | Filters headers/titles/noise |
| Sentence boundary | Period/Newline split | Clean label boundaries |

---

## Files Structure (Final)
```
telugu-ocr/
├── ground_truth/           # JSON ground truth (755 files)
├── source_images/images/   # 35k source page images
├── poppler/                # PDF utilities
├── scripts/
│   ├── build_ocr_dataset.py   # Main pipeline
│   └── requirements.txt
├── dataset/                # Output (after running)
│   ├── images/             # Paragraph crops (≥95% match)
│   ├── metadata.jsonl      # {image, text, match_score}
│   └── summary.json        # Stats
└── DEVELOPMENT_LOG.md      # This file
```

---

## Lessons Learned
1. **Simple > Clever:** Tesseract beat custom CV for text detection
2. **High threshold > Volume:** 95% match gives cleaner data than 60% with noise
3. **Unicode handling:** Always use encode/decode for non-ASCII paths
4. **Sentence boundaries:** Split by punctuation before matching, not after

---

## Next Steps
1. Run `build_dataset_full.py` on all 35k images (~6-8 hours)
2. Verify sample outputs
3. Upload to Hugging Face
4. Fine-tune TrOCR or similar model
