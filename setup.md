# 🚀 Quick Setup Guide

## Step 1: Copy These Files to Your Project

```
✅ app_with_stitching.py       (UPDATED - with preprocessing)
✅ image_preprocessing.py      (NEW)
✅ index.html                  (UPDATED - goes in templates/)
✅ requirements.txt            (UPDATED)
```

## Step 2: Install New Dependencies

```bash
pip install -r requirements.txt
```

The new preprocessing features use OpenCV functions you already have installed!

## Step 3: Make Sure You Have Templates Folder

```bash
mkdir -p templates
mv index.html templates/
```

## Step 4: Keep Your Existing Files

DON'T delete these - you still need them:
- ✅ ocr_easyocr_fast.py
- ✅ receipt_stitcher.py

## Step 5: Run the Server

```bash
python app_with_stitching.py
```

Open browser to: http://localhost:8000

## 🎯 What Changed?

### Backend (app_with_stitching.py)
- ✅ Added preprocessing imports
- ✅ New form parameters (auto_crop, deskew, denoise)
- ✅ Preprocessing applied before OCR
- ✅ Returns preprocessing info in response

### Frontend (index.html)
- ✅ Checkboxes for Crop, Deskew, Denoise
- ✅ Dropdown for denoise strength
- ✅ Shows preprocessing badges in results
- ✅ Updated button text with enabled features

### New Module (image_preprocessing.py)
- ✅ ImagePreprocessor class
- ✅ auto_crop() - removes borders
- ✅ deskew() - straightens tilted images
- ✅ denoise() - removes noise
- ✅ preprocess_receipt() - applies all steps

## 📦 File Sizes

- app_with_stitching.py: ~8 KB
- image_preprocessing.py: ~12 KB  
- index.html: ~15 KB
- Total new code: ~35 KB

## ⚡ Quick Test

```python
# Test preprocessing separately
python -c "
from image_preprocessing import preprocess_from_bytes
with open('receipt.jpg', 'rb') as f:
    img_bytes = f.read()
processed, info = preprocess_from_bytes(img_bytes, auto_crop=True, deskew=True, denoise=True)
print(f'Steps: {info[\"steps_applied\"]}')
"
```

## 🎨 UI Preview

When you open the web interface, you'll see:

1. **Upload Area** (same as before)
2. **NEW: Preprocessing Options** section with 3 checkboxes
3. **Process Button** (updates text based on selected features)
4. **Results** (now shows preprocessing badges)

## 🐛 Common Issues

**ImportError: cannot import name 'preprocess_from_bytes'**
→ Make sure image_preprocessing.py is in the same folder as app_with_stitching.py

**Template not found**
→ Make sure index.html is in the `templates/` folder

**Processing takes longer**
→ Normal! Preprocessing adds 2-5 seconds depending on features enabled

## 💡 Tips

- Start with just **denoise** enabled for faded receipts
- Use **deskew** for photos taken at angles
- Use **auto-crop** when receipt has colored background
- Enable all three for maximum quality (but slower processing)

---

Ready to roll! 🎉