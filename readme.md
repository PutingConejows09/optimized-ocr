# 📎 Advanced Receipt OCR with Image Enhancement

A production-ready FastAPI application for receipt OCR with automatic image stitching and advanced preprocessing features.

## ✨ Features

### Core Features
- 🔍 **Fast OCR** - EasyOCR optimized for 8-12 second processing
- 📎 **Auto-Stitching** - Automatically combine 2+ overlapping receipt photos
- 🖼️ **Beautiful UI** - Drag-and-drop interface with real-time preview

### NEW: Advanced Preprocessing
- 📐 **Auto-Crop** - Removes borders and background automatically
- 📏 **Deskew** - Straightens tilted receipts using Hough line detection
- 🧹 **Denoise** - Removes noise with adjustable strength (light/medium/strong)

## 📦 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Project Structure

```
your-project/
├── app_with_stitching.py      # Main FastAPI app
├── ocr_easyocr_fast.py         # Fast OCR engine
├── receipt_stitcher.py         # Image stitching module
├── image_preprocessing.py      # NEW: Preprocessing module
├── requirements.txt
└── templates/
    └── index.html              # Frontend UI
```

### 3. Required Files

Make sure you have these files from your original project:
- `ocr_easyocr_fast.py` - Your existing OCR engine
- `receipt_stitcher.py` - Your existing stitching module

## 🚀 Usage

### Start the Server

```bash
python app_with_stitching.py
```

Server runs at `http://localhost:8000`

### API Endpoints

#### Single Image Upload
```bash
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: Image file (required)
- auto_crop: boolean (optional, default: false)
- deskew: boolean (optional, default: false)
- denoise: boolean (optional, default: false)
- denoise_strength: 'light'|'medium'|'strong' (optional, default: 'medium')
```

#### Multiple Images Upload (with stitching)
```bash
POST /upload_multiple
Content-Type: multipart/form-data

Parameters:
- files: Image files (required, 2+ for stitching)
- auto_crop: boolean (optional)
- deskew: boolean (optional)
- denoise: boolean (optional)
- denoise_strength: string (optional)
```

### Example: Using cURL

**Single image with all preprocessing:**
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@receipt.jpg" \
  -F "auto_crop=true" \
  -F "deskew=true" \
  -F "denoise=true" \
  -F "denoise_strength=medium"
```

**Multiple images with stitching + preprocessing:**
```bash
curl -X POST http://localhost:8000/upload_multiple \
  -F "files=@receipt_part1.jpg" \
  -F "files=@receipt_part2.jpg" \
  -F "auto_crop=true" \
  -F "deskew=true"
```

## 🔧 Preprocessing Features Explained

### 📐 Auto-Crop
- Detects receipt edges using Canny edge detection
- Removes background and borders
- Adds configurable margin around detected receipt
- **Use when:** Receipt has dark/colored background or white borders

### 📏 Deskew
- Uses Hough line detection to find text orientation
- Automatically rotates image to straighten text
- Reports rotation angle in response
- **Use when:** Receipt photo is tilted or at an angle

### 🧹 Denoise
- Removes noise and artifacts using Non-Local Means Denoising
- Three strength levels:
  - **Light** - Fastest, minimal noise removal
  - **Medium** - Balanced (recommended)
  - **Strong** - Most aggressive, slower
- **Use when:** Receipt is faded, has artifacts, or poor print quality

## 📊 Response Format

```json
{
  "ocr_text": "Extracted text here...",
  "engine_used": "EasyOCR (Production-Fast)",
  "annotated_image": "base64_encoded_image",
  "stitched": false,
  "metrics": {
    "num_valid_words": 145,
    "avg_confidence": 87.5
  },
  "rating": "Excellent",
  "preprocessing": {
    "applied": true,
    "steps": ["denoise_medium", "deskew_2.35deg", "auto_crop"],
    "rotation_angle": 2.35
  },
  "stitching": {
    "num_images": 2,
    "method": "feature_matching",
    "stitched": true
  }
}
```

## 💡 Best Practices

### For Long Receipts
1. Take 2-3 overlapping photos (20-30% overlap between images)
2. Upload all photos together
3. Enable **deskew** if photos are taken at angles
4. Enable **denoise** if receipt is faded

### For Poor Quality Receipts
1. Enable all preprocessing: Crop + Deskew + Denoise
2. Use "medium" or "strong" denoising
3. Ensure good lighting when taking photos
4. Avoid shadows and glare

### For Fast Processing
1. Disable preprocessing if receipt quality is already good
2. Use "light" denoising if you need it
3. Single images process faster than stitching

## 🧪 Testing

### Test Preprocessing Module
```python
from image_preprocessing import preprocess_from_bytes

with open('receipt.jpg', 'rb') as f:
    image_bytes = f.read()

processed_img, info = preprocess_from_bytes(
    image_bytes,
    auto_crop=True,
    deskew=True,
    denoise=True,
    denoise_strength='medium'
)

print(f"Steps applied: {info['steps_applied']}")
print(f"Rotation: {info['rotation_angle']}°")
```

## ⚡ Performance

**Without Preprocessing:**
- Single image: ~8-12 seconds
- Stitched (2 images): ~15-20 seconds

**With All Preprocessing:**
- Single image: ~10-15 seconds (add ~2-3 sec)
- Stitched (2 images): ~18-25 seconds (add ~3-5 sec)

**Memory Usage:**
- Base: ~800MB
- Peak during processing: ~1.2GB
- Suitable for: AWS EC2 t3.micro (1GB RAM) with optimization

## 🐛 Troubleshooting

### "No contours found" - Auto-crop
- Receipt doesn't have clear edges
- Try without auto-crop or adjust image contrast

### "No lines detected" - Deskew
- Receipt is already straight
- Text is too sparse or image quality is poor

### Processing is slow
- Reduce image resolution before upload
- Disable unnecessary preprocessing steps
- Use "light" denoising instead of "strong"

### Out of memory
- Process images one at a time
- Reduce image size before processing
- Increase server RAM

## 📝 Notes

- Preprocessing happens BEFORE OCR
- Stitching happens BEFORE preprocessing (if multiple images)
- All preprocessing is optional and can be toggled individually
- Preprocessing improves OCR accuracy for poor quality receipts

## 🔒 Production Considerations

1. **Add authentication** if deploying publicly
2. **Rate limiting** to prevent abuse
3. **File size limits** (recommend max 10MB per image)
4. **CORS configuration** based on your frontend domain
5. **Error logging** for monitoring
6. **Health checks** at `/health` endpoint

## 📄 License

This is your project - license as you see fit!

## 🤝 Support

For issues or questions:
1. Check this README
2. Review the code comments
3. Test with different preprocessing combinations
4. Adjust parameters based on your specific receipt types

---

**Made with ❤️ by a vibe coder** 🎵