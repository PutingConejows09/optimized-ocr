# image_preprocessing.py
# Advanced image preprocessing: Cropping, Deskewing, Denoising

import cv2
import numpy as np
from typing import Tuple, Optional


class ImagePreprocessor:
    """
    Advanced receipt image preprocessing with:
    - Auto-cropping (removes borders/background)
    - Deskewing (straightens tilted receipts)
    - Denoising (removes noise/artifacts)
    """
    
    def __init__(self):
        pass
    
    def auto_crop(self, image: np.ndarray, margin: int = 10) -> np.ndarray:
        """
        Automatically crop receipt from background.
        Detects receipt edges and removes surrounding area.
        
        Args:
            image: Input image (BGR or grayscale)
            margin: Pixels to add around detected receipt
            
        Returns:
            Cropped image
        """
        print("📐 Auto-cropping receipt...")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            print("   ⚠️  No contours found, returning original")
            return image
        
        # Find largest contour (likely the receipt)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add margin and clamp to image bounds
        x = max(0, x - margin)
        y = max(0, y - margin)
        w = min(image.shape[1] - x, w + 2 * margin)
        h = min(image.shape[0] - y, h + 2 * margin)
        
        # Crop
        cropped = image[y:y+h, x:x+w]
        
        print(f"   ✅ Cropped from {image.shape[0]}x{image.shape[1]} to {cropped.shape[0]}x{cropped.shape[1]}")
        
        return cropped
    
    def deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Automatically straighten tilted receipt.
        Uses Hough line detection to find dominant angle.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            (deskewed_image, rotation_angle)
        """
        print("📏 Deskewing receipt...")
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(
            binary,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10
        )
        
        if lines is None or len(lines) == 0:
            print("   ⚠️  No lines detected, skipping deskew")
            return image, 0.0
        
        # Calculate angles of all lines
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            
            # Normalize angle to [-45, 45] range
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
            angles.append(angle)
        
        # Use median angle (more robust than mean)
        median_angle = np.median(angles)
        
        # Only deskew if angle is significant (> 0.5 degrees)
        if abs(median_angle) < 0.5:
            print(f"   ✅ Image already straight (angle: {median_angle:.2f}°)")
            return image, 0.0
        
        # Rotate image
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        
        # Calculate new image size to avoid cropping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        
        # Adjust rotation matrix for new size
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )
        
        print(f"   ✅ Rotated by {median_angle:.2f}° ({new_h}x{new_w}px)")
        
        return rotated, median_angle
    
    def denoise(self, image: np.ndarray, strength: str = 'medium') -> np.ndarray:
        """
        Remove noise from receipt image.
        
        Args:
            image: Input image (BGR or grayscale)
            strength: 'light', 'medium', or 'strong'
            
        Returns:
            Denoised image
        """
        print(f"🧹 Denoising (strength: {strength})...")
        
        # Denoising parameters
        params = {
            'light': {'h': 10, 'template': 7, 'search': 21},
            'medium': {'h': 15, 'template': 7, 'search': 21},
            'strong': {'h': 20, 'template': 9, 'search': 21}
        }
        
        p = params.get(strength, params['medium'])
        
        # Apply appropriate denoising based on image type
        if len(image.shape) == 3:
            # Color image - use fastNlMeansDenoisingColored
            denoised = cv2.fastNlMeansDenoisingColored(
                image,
                None,
                h=p['h'],
                hColor=p['h'],
                templateWindowSize=p['template'],
                searchWindowSize=p['search']
            )
        else:
            # Grayscale - use fastNlMeansDenoising
            denoised = cv2.fastNlMeansDenoising(
                image,
                None,
                h=p['h'],
                templateWindowSize=p['template'],
                searchWindowSize=p['search']
            )
        
        print(f"   ✅ Denoising complete")
        
        return denoised
    
    def preprocess_receipt(
        self,
        image: np.ndarray,
        auto_crop: bool = True,
        deskew: bool = True,
        denoise: bool = True,
        denoise_strength: str = 'medium'
    ) -> Tuple[np.ndarray, dict]:
        """
        Apply all preprocessing steps in optimal order.
        
        Args:
            image: Input image
            auto_crop: Whether to crop receipt
            deskew: Whether to straighten image
            denoise: Whether to remove noise
            denoise_strength: 'light', 'medium', or 'strong'
            
        Returns:
            (processed_image, info_dict)
        """
        print("\n" + "="*70)
        print("🔧 PREPROCESSING RECEIPT")
        print("="*70)
        
        processed = image.copy()
        info = {
            'original_size': image.shape[:2],
            'steps_applied': [],
            'rotation_angle': 0.0
        }
        
        # Step 1: Denoise (do this first to help edge detection)
        if denoise:
            processed = self.denoise(processed, denoise_strength)
            info['steps_applied'].append(f'denoise_{denoise_strength}')
        
        # Step 2: Deskew (before cropping for better edge detection)
        if deskew:
            processed, angle = self.deskew(processed)
            info['rotation_angle'] = angle
            if abs(angle) > 0.5:
                info['steps_applied'].append(f'deskew_{angle:.2f}deg')
        
        # Step 3: Auto-crop (last, to remove any borders added by rotation)
        if auto_crop:
            processed = self.auto_crop(processed)
            info['steps_applied'].append('auto_crop')
        
        info['final_size'] = processed.shape[:2]
        
        print("\n" + "="*70)
        print(f"✅ Preprocessing complete: {len(info['steps_applied'])} steps")
        print(f"   {' → '.join(info['steps_applied'])}")
        print("="*70 + "\n")
        
        return processed, info


def preprocess_from_bytes(
    image_bytes: bytes,
    auto_crop: bool = True,
    deskew: bool = True,
    denoise: bool = True,
    denoise_strength: str = 'medium'
) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to preprocess image from bytes.
    
    Args:
        image_bytes: Image data as bytes
        auto_crop: Whether to crop
        deskew: Whether to straighten
        denoise: Whether to denoise
        denoise_strength: Denoising strength
        
    Returns:
        (processed_image, info_dict)
    """
    # Decode image
    npimg = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Failed to decode image")
    
    # Preprocess
    preprocessor = ImagePreprocessor()
    return preprocessor.preprocess_receipt(
        image,
        auto_crop=auto_crop,
        deskew=deskew,
        denoise=denoise,
        denoise_strength=denoise_strength
    )