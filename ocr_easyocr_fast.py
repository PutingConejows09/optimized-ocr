# ocr_easyocr_fast.py
# PRODUCTION-OPTIMIZED: Fast OCR with good accuracy
# Target: 8-12 seconds per receipt (vs 20+ seconds for multi-strategy)

import cv2
import numpy as np
import os

_reader = None

def get_reader():
    """Lazy load EasyOCR reader"""
    global _reader
    if _reader is None:
        import easyocr
        import torch
        
        print("🔄 Loading EasyOCR model (fast mode)...")
        _reader = easyocr.Reader(
            ['en'], 
            gpu=False,
            verbose=False,
            quantize=True  # Faster inference
        )
        
        if hasattr(torch, 'set_num_threads'):
            torch.set_num_threads(2)  # Balance speed vs accuracy
        
        print("✅ EasyOCR loaded (production mode)")
    return _reader


def fast_preprocess(image_bytes):
    """
    FAST preprocessing optimized for production.
    Balances speed and quality - single best method.
    """
    npimg = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_GRAYSCALE)  # Start with grayscale
    if img is None:
        return None
    
    h, w = img.shape
    
    # OPTIMIZATION 1: Smaller target size (faster processing)
    TARGET_HEIGHT = 2200  # Reduced from 2800/3500
    MAX_HEIGHT = 2800     # Reduced from 3500
    
    if h < TARGET_HEIGHT:
        scale = TARGET_HEIGHT / h
        new_h = int(h * scale)
        new_w = int(w * scale)
        if new_h > MAX_HEIGHT:
            scale = MAX_HEIGHT / new_h
            new_h = int(new_h * scale)
            new_w = int(new_w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    elif h > MAX_HEIGHT:
        scale = MAX_HEIGHT / h
        img = cv2.resize(img, (int(w * scale), MAX_HEIGHT), interpolation=cv2.INTER_AREA)
    
    # OPTIMIZATION 2: Check if very faded, use appropriate method
    mean_brightness = np.mean(img)
    
    if mean_brightness > 180:
        # Very faded receipt - use Otsu (fast and effective)
        # Strong gamma first
        gamma = 0.45
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        img = cv2.LUT(img, table)
        
        # CLAHE (moderate for speed)
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Otsu binarization (very fast)
        img = cv2.GaussianBlur(img, (3, 3), 0)  # Smaller kernel for speed
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        if np.mean(img) < 127:
            img = cv2.bitwise_not(img)
    else:
        # Normal receipt - adaptive threshold (fast)
        # Light CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        
        # Adaptive threshold
        img = cv2.adaptiveThreshold(
            img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15,
            C=6
        )
    
    # OPTIMIZATION 3: Minimal morphology (for speed)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    # Convert to RGB for EasyOCR
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    return rgb


def run_ocr_fast(img):
    """
    Run EasyOCR with parameters optimized for SPEED.
    Target: 6-8 seconds for OCR (vs 12-15 seconds in multi-strategy)
    """
    reader = get_reader()
    
    # OPTIMIZATION 4: Faster EasyOCR parameters
    results = reader.readtext(
        img,
        paragraph=False,
        width_ths=0.7,
        contrast_ths=0.1,
        adjust_contrast=0.4,
        text_threshold=0.25,    # Higher = faster (fewer false positives)
        low_text=0.2,           # Higher = faster
        link_threshold=0.5,
        mag_ratio=2.0,          # Lower = faster (was 2.5-3.0)
        canvas_size=3000,       # Smaller = faster (was 3500-4000)
        height_ths=0.7,
        add_margin=0.1,
        slope_ths=0.2,
        min_size=12,            # Larger = faster (skip tiny noise)
        batch_size=10           # Process in batches for speed
    )
    
    # OPTIMIZATION 5: Aggressive filtering (faster post-processing)
    word_data = []
    annotated = img.copy()
    
    for (bbox, text, conf) in results:
        # Higher confidence threshold = fewer false positives = faster
        if conf < 0.25:  # Higher than 0.20
            continue
        
        # Skip very short fragments quickly
        text = text.strip()
        if len(text) <= 1:
            continue
        
        if len(text) == 2 and conf < 0.5:
            continue
        
        top_left = tuple(map(int, bbox[0]))
        bottom_right = tuple(map(int, bbox[2]))
        
        word_details = {
            'text': text,
            'confidence': round(conf * 100, 2),
            'left': top_left[0], 
            'top': top_left[1], 
            'width': bottom_right[0] - top_left[0], 
            'height': bottom_right[1] - top_left[1]
        }
        word_data.append(word_details)
        
        # Simple coloring for speed
        color = (0, 255, 0) if conf >= 0.7 else (0, 255, 255) if conf >= 0.5 else (0, 165, 255)
        cv2.rectangle(annotated, top_left, bottom_right, color, 2)
    
    # Sort by reading order
    word_data.sort(key=lambda w: (w['top'], w['left']))
    
    return word_data, annotated


def reconstruct_text_fast(words):
    """Fast text reconstruction"""
    if not words:
        return "No text detected"
    
    lines = []
    current_line = []
    last_top = words[0]['top']
    last_height = words[0]['height']
    
    for word in words:
        if word['top'] > last_top + (last_height * 0.6):
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word['text']]
            last_top = word['top']
            last_height = word['height']
        else:
            current_line.append(word['text'])
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


def run_easyocr(image_bytes, image_path=None, base_log_dir="log_fast"):
    """
    PRODUCTION-OPTIMIZED OCR
    Target: 8-12 seconds total (vs 20+ for multi-strategy)
    
    Trade-offs:
    - Speed: 2-3x faster
    - Accuracy: ~85-90% of multi-strategy quality
    - Good enough for production with many users
    """
    try:
        import time
        start_time = time.time()
        
        print("\n" + "="*70)
        print("⚡ PRODUCTION-OPTIMIZED OCR (FAST MODE)")
        print("   Target: 8-12 seconds | Good accuracy")
        print("="*70 + "\n")
        
        # Preprocessing: ~2-3 seconds
        preprocess_start = time.time()
        img_processed = fast_preprocess(image_bytes)
        preprocess_time = time.time() - preprocess_start
        print(f"✅ Preprocessing: {preprocess_time:.1f}s")
        
        if img_processed is None:
            return [], None, None
        
        # Save preprocessing result (optional)
        if image_path:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            log_dir = os.path.join(base_log_dir, image_name)
        else:
            log_dir = base_log_dir
        
        os.makedirs(log_dir, exist_ok=True)
        
        # OCR: ~6-8 seconds
        ocr_start = time.time()
        words, annotated = run_ocr_fast(img_processed)
        ocr_time = time.time() - ocr_start
        print(f"✅ OCR detection: {ocr_time:.1f}s")
        
        # Post-processing: ~0.5 seconds
        post_start = time.time()
        
        if words:
            avg_conf = np.mean([w['confidence'] for w in words])
            print(f"✅ Detected {len(words)} words, avg confidence: {avg_conf:.1f}%")
        
        # Reconstruct text
        ocr_text = reconstruct_text_fast(words)
        
        # Save results (optional - can skip in production for speed)
        if image_path:
            cv2.imwrite(os.path.join(log_dir, "fast_processed.jpg"), img_processed)
            cv2.imwrite(os.path.join(log_dir, "fast_annotated.jpg"), annotated)
            
            with open(os.path.join(log_dir, "ocr_text.txt"), "w", encoding="utf-8") as f:
                f.write(ocr_text)
        
        # Create evaluation
        evaluation = None
        if words:
            img_height, img_width = img_processed.shape[:2]
            total_area = sum(w['width'] * w['height'] for w in words)
            image_area = img_width * img_height
            area_ratio = (total_area / image_area) * 100
            avg_confidence = np.mean([w['confidence'] for w in words])
            
            evaluation = {
                'metrics': {
                    'num_valid_words': len(words),
                    'avg_confidence': round(avg_confidence, 2),
                    'area_ratio_pct': round(area_ratio, 2)
                },
                'rating': 'Excellent' if avg_confidence > 80 and len(words) > 60 else
                         'Good' if avg_confidence > 70 and len(words) > 40 else
                         'Reasonable' if avg_confidence > 60 and len(words) > 25 else 'Poor'
            }
        
        post_time = time.time() - post_start
        print(f"✅ Post-processing: {post_time:.1f}s")
        
        total_time = time.time() - start_time
        print(f"\n⏱️  TOTAL TIME: {total_time:.1f}s")
        
        if evaluation:
            print(f"📊 Results: {len(words)} words, {evaluation['metrics']['avg_confidence']:.1f}% confidence")
            print(f"   Rating: {evaluation['rating']}")
        
        # Cleanup
        del img_processed
        import gc
        gc.collect()
        
        return words, evaluation, annotated
        
    except Exception as e:
        print(f"❌ Fast OCR failed: {e}")
        import traceback
        print(traceback.format_exc())
        return [], None, None