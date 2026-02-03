---
language:
- te
license: mit
task_categories:
- image-to-text
- object-detection
tags:
- ocr
- telugu
- chandamama
- trocr
- document-understanding
pretty_name: Chandamama Telugu OCR Dataset
size_categories:
- 1k<n<10k
---

# 📚 Chandamama Telugu OCR Dataset (1947-2012)

A large-scale, high-quality Optical Character Recognition (OCR) dataset derived from the historic **Chandamama** children's magazine archives (Telugu edition). This dataset is specifically curated for fine-tuning TrOCR and other Vision-Language models for Indian language text recognition.

## 📊 Dataset Statistics

- **Total Examples**: 9,640 image-text pairs
- **Time Period**: 66 years (July 1947 - Dec 2012)
- **Source**: Scanned PDF pages from Chandamama archives
- **Language**: Telugu (తెలుగు)
- **Image Format**: Grayscale JPEG (optimized for efficiency)
- **Total Size**: ~4.2 GB

## 📂 Dataset Structure

Each row in the dataset contains the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `image` | `PIL.Image` | The converted grayscale image of the story page. |
| `text` | `string` | The ground truth Telugu text corresponding to the page content. |
| `year` | `int64` | The publication year (e.g., 1947, 2005). |
| `source_pdf` | `string` | The original PDF filename (e.g., `చందమామ 1947 07.pdf`). |
| `title` | `string` | The title of the story (if available). |
| `story_id` | `string` | Unique identifier for the story. |

## 🛠️ Creation Process

1.  **Source Collection**: 790+ PDFs corresponding to monthly magazines from 1947 to 2012 were collected.
2.  **Metadata Alignment**: Each PDF was paired with a JSON metadata file containing story boundaries and ground truth text.
3.  **Extraction**: Pages corresponding to stories were extracted as high-resolution PNGs.
4.  **Optimization**:
    *   Converted to **Grayscale** (L mode) to reduce color noise.
    *   Compressed to **JPEG (Quality 85)** to optimize storage (size reduced by ~94% without OCR quality loss).
    *   Final size: ~4.2 GB.
5.  **Validation**: Strict matching ensured every image has corresponding text. Unmatched pages were discarded.

## 🚀 Usage Guide

### Loading in Python (Hugging Face)

```python
from datasets import load_dataset

dataset = load_dataset("Divs0910/Telugu-OCR-Dataset-001")

# View a sample
print(dataset['train'][0]['text'])
dataset['train'][0]['image'].show()
```

### Fine-tuning TrOCR (Google Colab)

This dataset is optimized for fine-tuning **Microsoft's TrOCR** (Transformer-based Optical Character Recognition).
- **Input**: `image` (resized to 384x384 or similar)
- **Output**: `text` (tokenized Telugu sequence)

**Recommended Model**: `microsoft/trocr-base-printed`

## 📝 License

This dataset is intended for research and educational purposes. The original content (Chandamama stories) belongs to their respective copyright holders.

## 👨‍💻 Curated By
*   **Divyansh Teja** (2026)
*   *Part of the Telugu OCR Project*
