# app_with_stitching.py
# FastAPI app with automatic receipt stitching and preprocessing support

import os
import json
import cv2
import numpy as np
import uvicorn
import base64
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import OCR (use fast version for production)
from ocr_easyocr_fast import run_easyocr

# Import stitcher and preprocessor
from receipt_stitcher import stitch_from_bytes
from image_preprocessing import preprocess_from_bytes

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# S3 handler
s3_handler = None
print("⚠️  S3 Handler disabled for local testing")
print("🎯 OCR Engine: EasyOCR (Production-Fast)")
print("📎 Features: Stitching, Cropping, Deskewing, Denoising")

def reconstruct_text_layout(ocr_data):
    """Reconstructs text from OCR data by preserving spatial layout."""
    if not ocr_data:
        return "No text found."

    # Group words into lines
    ocr_data.sort(key=lambda w: w['top'])
    lines = []
    
    if ocr_data:
        current_line = [ocr_data[0]]
        ref_height = np.median([w['height'] for w in ocr_data[:min(10, len(ocr_data))]])
        
        for word in ocr_data[1:]:
            line_y_center = sum(w['top'] + w['height'] / 2 for w in current_line) / len(current_line)
            word_y_center = word['top'] + word['height'] / 2
            
            if abs(word_y_center - line_y_center) > ref_height * 0.8:
                lines.append(current_line)
                current_line = [word]
            else:
                current_line.append(word)
        lines.append(current_line)

    # Reconstruct each line
    output_lines = []
    for line in lines:
        line.sort(key=lambda w: w['left'])
        
        if not line:
            continue
        
        line_text = " ".join(w['text'] for w in line)
        output_lines.append(line_text)

    return "\n".join(output_lines)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "status": "healthy" if mem.percent < 85 else "degraded",
            "ocr_engine": "EasyOCR (Production-Fast)",
            "features": ["stitching", "cropping", "deskewing", "denoising", "fast_ocr"],
            "memory_percent": mem.percent,
            "memory_available_mb": round(mem.available / 1024 / 1024, 1),
            "s3_enabled": s3_handler is not None
        }
    except ImportError:
        return {
            "status": "healthy",
            "ocr_engine": "EasyOCR (Production-Fast)",
            "features": ["stitching", "cropping", "deskewing", "denoising", "fast_ocr"],
            "s3_enabled": s3_handler is not None
        }

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    auto_crop: bool = Form(False),
    deskew: bool = Form(False),
    denoise: bool = Form(False),
    denoise_strength: str = Form('medium')
):
    """
    Single file upload endpoint with preprocessing options.
    """
    try:
        image_bytes = await file.read()
        
        print("\n" + "="*60)
        print("🚀 Single Image Upload")
        print("="*60)
        
        # Apply preprocessing if requested
        preprocessing_info = None
        if auto_crop or deskew or denoise:
            print("\n🔧 Applying preprocessing...")
            processed_img, preprocessing_info = preprocess_from_bytes(
                image_bytes,
                auto_crop=auto_crop,
                deskew=deskew,
                denoise=denoise,
                denoise_strength=denoise_strength
            )
            
            # Encode back to bytes
            _, buffer = cv2.imencode('.jpg', processed_img)
            image_bytes = buffer.tobytes()
        
        # Run OCR
        ocr_data, evaluation, annotated_img = run_easyocr(image_bytes)
        
        # Prepare response
        response = prepare_response(ocr_data, evaluation, annotated_img, stitched=False)
        
        if preprocessing_info:
            response['preprocessing'] = {
                'applied': True,
                'steps': preprocessing_info['steps_applied'],
                'rotation_angle': preprocessing_info.get('rotation_angle', 0)
            }
        
        return response
        
    except Exception as e:
        import traceback
        print(f"Error processing upload: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload_multiple")
async def upload_multiple_files(
    files: List[UploadFile] = File(...),
    auto_crop: bool = Form(False),
    deskew: bool = Form(False),
    denoise: bool = Form(False),
    denoise_strength: str = Form('medium')
):
    """
    Multiple file upload with automatic stitching and preprocessing.
    """
    try:
        print("\n" + "="*60)
        print(f"🚀 Multiple Image Upload: {len(files)} files")
        print("="*60)
        
        if len(files) == 0:
            return JSONResponse(status_code=400, content={"error": "No files provided"})
        
        # Read all images
        image_bytes_list = []
        for file in files:
            image_bytes = await file.read()
            image_bytes_list.append(image_bytes)
        
        # Stitch if multiple images
        preprocessing_info = None
        if len(image_bytes_list) > 1:
            print(f"\n📎 Stitching {len(image_bytes_list)} images...")
            
            # Stitch images
            stitched_img, stitch_info = stitch_from_bytes(image_bytes_list)
            
            print(f"✅ Stitching complete: {stitch_info['method']}")
            print(f"   Output: {stitched_img.shape[0]}x{stitched_img.shape[1]}px")
            
            # Apply preprocessing to stitched image if requested
            if auto_crop or deskew or denoise:
                print("\n🔧 Applying preprocessing to stitched image...")
                from image_preprocessing import ImagePreprocessor
                preprocessor = ImagePreprocessor()
                stitched_img, preprocessing_info = preprocessor.preprocess_receipt(
                    stitched_img,
                    auto_crop=auto_crop,
                    deskew=deskew,
                    denoise=denoise,
                    denoise_strength=denoise_strength
                )
            
            # Encode stitched image to bytes
            _, buffer = cv2.imencode('.jpg', stitched_img)
            stitched_bytes = buffer.tobytes()
            
            # Run OCR on stitched image
            print(f"\n🔍 Running OCR on stitched image...")
            ocr_data, evaluation, annotated_img = run_easyocr(stitched_bytes)
            
            # Prepare response with stitching info
            response = prepare_response(ocr_data, evaluation, annotated_img, stitched=True)
            response['stitching'] = {
                'num_images': stitch_info['num_images'],
                'method': stitch_info['method'],
                'stitched': True
            }
            
            if preprocessing_info:
                response['preprocessing'] = {
                    'applied': True,
                    'steps': preprocessing_info['steps_applied'],
                    'rotation_angle': preprocessing_info.get('rotation_angle', 0)
                }
            
            return response
        else:
            # Single image - process normally
            image_bytes = image_bytes_list[0]
            
            # Apply preprocessing if requested
            if auto_crop or deskew or denoise:
                print("\n🔧 Applying preprocessing...")
                processed_img, preprocessing_info = preprocess_from_bytes(
                    image_bytes,
                    auto_crop=auto_crop,
                    deskew=deskew,
                    denoise=denoise,
                    denoise_strength=denoise_strength
                )
                
                # Encode back to bytes
                _, buffer = cv2.imencode('.jpg', processed_img)
                image_bytes = buffer.tobytes()
            
            ocr_data, evaluation, annotated_img = run_easyocr(image_bytes)
            response = prepare_response(ocr_data, evaluation, annotated_img, stitched=False)
            
            if preprocessing_info:
                response['preprocessing'] = {
                    'applied': True,
                    'steps': preprocessing_info['steps_applied'],
                    'rotation_angle': preprocessing_info.get('rotation_angle', 0)
                }
            
            return response
        
    except Exception as e:
        import traceback
        print(f"Error processing upload: {traceback.format_exc()}")
        return JSONResponse(status_code=500, content={"error": str(e)})

def prepare_response(ocr_data, evaluation, annotated_img, stitched=False):
    """Prepare standard response"""
    final_data = ocr_data if ocr_data else []
    final_annotated_img = annotated_img if annotated_img is not None else np.zeros((100, 100, 3), dtype=np.uint8)
    
    ocr_text = reconstruct_text_layout(final_data)
    
    # Encode annotated image
    _, buffer = cv2.imencode('.jpg', final_annotated_img)
    annotated_bytes = buffer.tobytes()
    base64_img = base64.b64encode(annotated_bytes).decode('utf-8')
    
    # Prepare response
    response_data = {
        "ocr_text": ocr_text,
        "engine_used": "EasyOCR (Production-Fast)",
        "annotated_image": base64_img,
        "stitched": stitched
    }
    
    # Add metrics
    if evaluation and 'metrics' in evaluation:
        response_data["metrics"] = evaluation['metrics']
        response_data["rating"] = evaluation.get('rating', 'Unknown')
    
    # Cleanup
    del annotated_bytes
    import gc
    gc.collect()

    return JSONResponse(content=response_data)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    print(f"\n🚀 Starting server on port {port}")
    print(f"💡 OCR Engine: EasyOCR (Production-Fast)")
    print(f"📎 Features: Stitching, Cropping, Deskewing, Denoising")
    print(f"🔧 Optimized for: T3 Micro (1GB RAM)")
    print(f"\nEndpoints:")
    print(f"  POST /upload          - Single image with preprocessing")
    print(f"  POST /upload_multiple - Multiple images (auto-stitch + preprocessing)")
    print()
    uvicorn.run(app, host="0.0.0.0", port=port)