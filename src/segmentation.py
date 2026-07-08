# src/segmentation.py
# Retinal layer segmentation for OCT images

import cv2
import numpy as np
from scipy.ndimage import label, binary_opening, binary_closing

class OCTSegmenter:
    """Segment retinal layers and detect abnormalities in OCT images."""
    
    @staticmethod
    def adaptive_threshold(image, block_size=51, c=10):
        """
        Apply adaptive thresholding to detect retinal boundaries.
        
        Args:
            image: Input preprocessed OCT image (uint8)
            block_size: Size of neighborhood (odd number)
            c: Constant subtracted from mean
            
        Returns:
            binary_image: Binary image with retinal layers
        """
        binary = cv2.adaptiveThreshold(
            image, 
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=block_size,
            C=c
        )
        return binary
    
    @staticmethod
    def detect_ilm_boundary(image):
        """
        Detect ILM (Internal Limiting Membrane) - top retinal boundary.
        
        Args:
            image: Preprocessed OCT image
            
        Returns:
            ilm_points: Y-coordinates of ILM for each X position
        """
        # Apply threshold
        binary = OCTSegmenter.adaptive_threshold(image)
        
        # Find top boundary (ILM)
        ilm_points = []
        for x in range(binary.shape[1]):
            column = binary[:, x]
            # Find first white pixel from top
            white_pixels = np.where(column > 0)[0]
            if len(white_pixels) > 0:
                ilm_points.append(white_pixels[0])
            else:
                ilm_points.append(0)
        
        return np.array(ilm_points)
    
    @staticmethod
    def detect_rpe_boundary(image):
        """
        Detect RPE (Retinal Pigment Epithelium) - bottom retinal boundary.
        
        Args:
            image: Preprocessed OCT image
            
        Returns:
            rpe_points: Y-coordinates of RPE for each X position
        """
        # Apply threshold
        binary = OCTSegmenter.adaptive_threshold(image)
        
        # Find bottom boundary (RPE)
        rpe_points = []
        for x in range(binary.shape[1]):
            column = binary[:, x]
            # Find last white pixel from bottom
            white_pixels = np.where(column > 0)[0]
            if len(white_pixels) > 0:
                rpe_points.append(white_pixels[-1])
            else:
                rpe_points.append(image.shape[0] - 1)
        
        return np.array(rpe_points)
    
    @staticmethod
    def detect_fluid(image, threshold=100):
        """
        Detect abnormal fluid (dark regions) within retina.
        Indicates DME (Diabetic Macular Edema) or other pathology.
        
        Args:
            image: Preprocessed OCT image
            threshold: Intensity threshold for dark fluid
            
        Returns:
            fluid_mask: Binary mask of detected fluid regions
        """
        # Dark fluid regions have low intensity
        fluid_mask = image < threshold
        
        # Remove noise with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fluid_mask = cv2.morphologyEx(fluid_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        
        return fluid_mask.astype(bool)
    
    @staticmethod
    def segment_retina(image):
        """
        Complete retinal segmentation pipeline.
        
        Args:
            image: Preprocessed OCT image (uint8)
            
        Returns:
            segmentation_dict: Dictionary with ILM, RPE, fluid mask, retinal thickness
        """
        # Detect boundaries
        ilm = OCTSegmenter.detect_ilm_boundary(image)
        rpe = OCTSegmenter.detect_rpe_boundary(image)
        fluid = OCTSegmenter.detect_fluid(image)
        
        # Calculate retinal thickness
        thickness = np.maximum(rpe - ilm, 0)  # Ensure non-negative
        mean_thickness = np.mean(thickness)
        
        # Count fluid regions
        fluid_volume = np.sum(fluid)
        
        return {
            'ilm': ilm,
            'rpe': rpe,
            'fluid_mask': fluid,
            'thickness': thickness,
            'mean_thickness': mean_thickness,
            'fluid_volume': fluid_volume,
            'has_fluid': fluid_volume > 0
        }
