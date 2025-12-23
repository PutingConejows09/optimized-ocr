# receipt_stitcher.py
# Automatic stitching for multi-part receipt images

import cv2
import numpy as np
from typing import List, Tuple, Optional

class ReceiptStitcher:
    """
    Stitches multiple receipt images into a single long image.
    Handles 2+ images with automatic overlap detection.
    """
    
    def __init__(self, overlap_threshold: float = 0.3):
        """
        Args:
            overlap_threshold: Minimum overlap ratio to consider images as parts (0.3 = 30%)
        """
        self.overlap_threshold = overlap_threshold
    
    def stitch_receipts(self, images: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
        """
        Stitch multiple receipt images into one.
        
        Args:
            images: List of images as numpy arrays (BGR format)
            
        Returns:
            (stitched_image, info_dict)
            - stitched_image: Combined image
            - info_dict: Metadata about stitching
        """
        if not images or len(images) == 0:
            raise ValueError("No images provided")
        
        if len(images) == 1:
            return images[0], {
                'num_images': 1,
                'stitched': False,
                'method': 'single_image'
            }
        
        print(f"\n{'='*70}")
        print(f"📎 RECEIPT STITCHING")
        print(f"   Images to stitch: {len(images)}")
        print(f"{'='*70}\n")
        
        # Try automatic stitching first
        result = self._auto_stitch(images)
        
        if result is not None:
            return result
        
        # Fallback: Simple vertical concatenation
        print("⚠️  Auto-stitch failed, using vertical concatenation fallback")
        return self._simple_vertical_concat(images)
    
    def _auto_stitch(self, images: List[np.ndarray]) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Automatic stitching using feature matching.
        Works well when images have overlap.
        """
        try:
            # Sort images by likely receipt order (top to bottom)
            sorted_images = self._sort_by_position(images)
            
            # Start with first image
            stitched = sorted_images[0].copy()
            stitch_info = {
                'num_images': len(images),
                'stitched': True,
                'method': 'feature_matching',
                'overlaps': []
            }
            
            print(f"🔍 Stitching {len(sorted_images)} images...")
            
            # Stitch each subsequent image
            for i in range(1, len(sorted_images)):
                print(f"\n   Stitching image {i+1}/{len(sorted_images)}...")
                
                next_img = sorted_images[i]
                
                # Find overlap and stitch
                stitched_new, overlap_info = self._stitch_pair(stitched, next_img)
                
                if stitched_new is not None:
                    stitched = stitched_new
                    stitch_info['overlaps'].append(overlap_info)
                    print(f"   ✅ Overlap: {overlap_info['overlap_pixels']}px")
                else:
                    # Fallback: just append below
                    print(f"   ⚠️  No overlap found, appending below")
                    stitched = self._append_vertical(stitched, next_img)
                    stitch_info['overlaps'].append({'overlap_pixels': 0, 'method': 'append'})
            
            print(f"\n✅ Stitching complete: {stitched.shape[0]}x{stitched.shape[1]}px")
            return stitched, stitch_info
            
        except Exception as e:
            print(f"❌ Auto-stitch failed: {e}")
            return None
    
    def _stitch_pair(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[Optional[np.ndarray], dict]:
        """
        Stitch two images with overlap detection.
        Assumes img2 is below img1 (receipt continues downward).
        """
        # Get bottom portion of img1 and top portion of img2
        overlap_height = min(img1.shape[0] // 3, img2.shape[0] // 3, 500)  # Max 500px overlap
        
        bottom_img1 = img1[-overlap_height:, :]
        top_img2 = img2[:overlap_height, :]
        
        # Find matching features
        matches, offset = self._find_overlap(bottom_img1, top_img2)
        
        if matches is not None and len(matches) >= 10:
            # Good overlap found
            overlap_pixels = overlap_height - offset
            
            # Blend the overlapping region
            stitched = self._blend_images(img1, img2, overlap_pixels)
            
            return stitched, {
                'overlap_pixels': overlap_pixels,
                'matches': len(matches),
                'method': 'feature_match'
            }
        else:
            # No good overlap - just append
            return None, {'overlap_pixels': 0, 'method': 'no_match'}
    
    def _find_overlap(self, img1: np.ndarray, img2: np.ndarray) -> Tuple[Optional[list], int]:
        """
        Find overlap between two images using ORB features.
        Returns (matches, vertical_offset).
        """
        try:
            # Convert to grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
            
            # Use ORB (faster than SIFT, free to use)
            orb = cv2.ORB_create(nfeatures=1000)
            
            # Find keypoints and descriptors
            kp1, des1 = orb.detectAndCompute(gray1, None)
            kp2, des2 = orb.detectAndCompute(gray2, None)
            
            if des1 is None or des2 is None:
                return None, 0
            
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            
            if len(matches) < 10:
                return None, 0
            
            # Sort by distance
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Calculate average vertical offset
            offsets = []
            for match in matches[:20]:  # Use top 20 matches
                pt1 = kp1[match.queryIdx].pt
                pt2 = kp2[match.trainIdx].pt
                offsets.append(pt1[1] - pt2[1])
            
            avg_offset = int(np.median(offsets))
            
            return matches, avg_offset
            
        except Exception as e:
            print(f"   ⚠️  Feature matching failed: {e}")
            return None, 0
    
    def _blend_images(self, img1: np.ndarray, img2: np.ndarray, overlap: int) -> np.ndarray:
        """
        Blend two images with given overlap pixels.
        Creates smooth transition in overlap region.
        """
        if overlap <= 0:
            return self._append_vertical(img1, img2)
        
        # Ensure same width
        if img1.shape[1] != img2.shape[1]:
            target_width = max(img1.shape[1], img2.shape[1])
            img1 = self._resize_width(img1, target_width)
            img2 = self._resize_width(img2, target_width)
        
        # Calculate dimensions
        h1, w = img1.shape[:2]
        h2 = img2.shape[0]
        
        # Clamp overlap
        overlap = min(overlap, h1 // 2, h2 // 2)
        
        # Create output image
        total_height = h1 + h2 - overlap
        stitched = np.zeros((total_height, w, 3), dtype=np.uint8)
        
        # Copy top part of img1
        stitched[:h1-overlap, :] = img1[:h1-overlap, :]
        
        # Blend overlap region
        overlap_start = h1 - overlap
        for i in range(overlap):
            alpha = i / overlap  # 0 to 1
            stitched[overlap_start + i, :] = (
                img1[h1 - overlap + i, :] * (1 - alpha) +
                img2[i, :] * alpha
            ).astype(np.uint8)
        
        # Copy bottom part of img2
        stitched[h1:, :] = img2[overlap:, :]
        
        return stitched
    
    def _append_vertical(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """
        Simply concatenate images vertically (no overlap).
        """
        # Ensure same width
        if img1.shape[1] != img2.shape[1]:
            target_width = max(img1.shape[1], img2.shape[1])
            img1 = self._resize_width(img1, target_width)
            img2 = self._resize_width(img2, target_width)
        
        return np.vstack([img1, img2])
    
    def _resize_width(self, img: np.ndarray, target_width: int) -> np.ndarray:
        """Resize image to target width, maintaining aspect ratio."""
        h, w = img.shape[:2]
        if w == target_width:
            return img
        
        scale = target_width / w
        new_h = int(h * scale)
        return cv2.resize(img, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
    
    def _simple_vertical_concat(self, images: List[np.ndarray]) -> Tuple[np.ndarray, dict]:
        """
        Fallback: Simple vertical concatenation without overlap detection.
        """
        print("📎 Using simple vertical concatenation")
        
        # Sort by likely position
        sorted_images = self._sort_by_position(images)
        
        # Find max width
        max_width = max(img.shape[1] for img in sorted_images)
        
        # Resize all to same width
        resized = [self._resize_width(img, max_width) for img in sorted_images]
        
        # Concatenate
        stitched = np.vstack(resized)
        
        return stitched, {
            'num_images': len(images),
            'stitched': True,
            'method': 'simple_concat',
            'overlaps': []
        }
    
    def _sort_by_position(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Sort images by likely position in receipt (top to bottom).
        Uses average brightness - receipt top is often header (darker text).
        """
        # Calculate average brightness for each image's top portion
        brightness_scores = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            top_portion = gray[:gray.shape[0]//4, :]  # Top 25%
            brightness = np.mean(top_portion)
            brightness_scores.append(brightness)
        
        # Sort by brightness (darker first, usually header)
        sorted_indices = np.argsort(brightness_scores)
        return [images[i] for i in sorted_indices]


def stitch_from_files(image_paths: List[str]) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to stitch images from file paths.
    
    Args:
        image_paths: List of image file paths
        
    Returns:
        (stitched_image, info_dict)
    """
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        images.append(img)
    
    stitcher = ReceiptStitcher()
    return stitcher.stitch_receipts(images)


def stitch_from_bytes(image_bytes_list: List[bytes]) -> Tuple[np.ndarray, dict]:
    """
    Convenience function to stitch images from bytes.
    
    Args:
        image_bytes_list: List of image data as bytes
        
    Returns:
        (stitched_image, info_dict)
    """
    images = []
    for image_bytes in image_bytes_list:
        npimg = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from bytes")
        images.append(img)
    
    stitcher = ReceiptStitcher()
    return stitcher.stitch_receipts(images)