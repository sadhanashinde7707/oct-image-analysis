# src/preprocessing.py
# Image preprocessing module for OCT images

import cv2
import numpy as np
from scipy.ndimage import median_filter

class OCTPreprocessor:
    """Preprocessing pipeline for OCT images."""
    
    @staticmethod
    def normalize(image):
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def denormalize(image):
        """Convert normalized image back to [0, 255] uint8."""
        return np.clip(image * 255, 0, 255).astype(np.uint8)
    
    @staticmethod
    def apply_clahe(image, clip_limit=2.0, tile_size=8):
        """Apply Contrast Limited Adaptive Histogram Equalization."""
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(image)
        return enhanced
    
    @staticmethod
    def denoise_median(image, kernel_size=5):
        """Apply median filtering to remove speckle noise."""
        denoised = median_filter(image, size=kernel_size)
        return denoised
    
    @staticmethod
    def denoise_bilateral(image, diameter=9, sigma_color=75, sigma_space=75):
        """Apply bilateral filtering (edge-preserving denoising)."""
        denoised = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
        return denoised
    
    @staticmethod
    def preprocess_pipeline(image, apply_clahe=True, denoise_method='bilateral'):
        """Full preprocessing pipeline for OCT images."""
        if apply_clahe:
            image = OCTPreprocessor.apply_clahe(image, clip_limit=2.0, tile_size=8)
        
        if denoise_method == 'bilateral':
            image = OCTPreprocessor.denoise_bilateral(image)
        elif denoise_method == 'median':
            image = OCTPreprocessor.denoise_median(image)
        
        return image
