# Telugu OCR Dataset

A high-quality Telugu OCR training dataset built from ~35,000 Chandamama magazine page images with paragraph-level ground truth text.

## Dataset Statistics
- **Source**: Chandamama magazine archives (1947-2012)
- **Pages**: ~35,000 grayscale page images
- **Output**: Paragraph-level crops with fuzzy-matched ground truth text
- **Quality**: Only ≥95% confidence matches retained

## Project Structure
```
telugu-ocr/
├── ground_truth/           # JSON files with story content
├── source_images/images/   # Original page images (not in git)
├── scripts/
│   ├── build_ocr_dataset.py   # Main pipeline
│   ├── upload_to_hf.py        # Hugging Face uploader
│   └── requirements.txt
├── dataset/                # Generated output (after running pipeline)
│   ├── images/             # Paragraph crops
│   ├── metadata.jsonl      # {image, text, match_score}
│   └── summary.json
└── DEVELOPMENT_LOG.md      # Development history
```

## Pipeline Overview
1. **Tesseract Extraction**: Detect paragraphs (4+ lines) using Tesseract OCR
2. **Fuzzy Matching**: Match Tesseract output to ground truth using rapidfuzz
3. **Quality Filter**: Keep only ≥95% match score pairs
4. **Output**: Clean paragraph crops + aligned ground truth text

## Usage

### Prerequisites
```bash
pip install -r scripts/requirements.txt
```
Also install Tesseract OCR with Telugu language:
- Windows: Download from [UB-Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
- Add `tel.traineddata` to `tessdata` folder

### Build Dataset
```bash
python scripts/build_ocr_dataset.py
```
This will:
- Process all images from `source_images/`
- Extract paragraph crops
- Match with ground truth
- Save to `dataset/` folder

### Upload to Hugging Face
```bash
python scripts/upload_to_hf.py
```

## Dataset on Hugging Face
🤗 [Dataset Link](https://huggingface.co/datasets/DIVYANSH-TEJA-09/telugu-ocr-paragraphs)

## License
The dataset is derived from Chandamama magazine archives. Use for research and educational purposes.

## Citation
```bibtex
@dataset{telugu_ocr_2024,
  title={Telugu OCR Paragraph Dataset},
  author={DIVYANSH-TEJA-09},
  year={2024},
  publisher={Hugging Face}
}
```
