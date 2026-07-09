# src/features.py
import numpy as np
import cv2
from scipy.stats import entropy
from src.preprocessing import OCTPreprocessor
from src.segmentation import OCTSegmenter

class FeatureExtractor:
    @staticmethod
    def extract_thickness_features(segmentation):
        thickness = segmentation['thickness']
        result = {
            'mean_thickness': float(np.mean(thickness)),
            'std_thickness': float(np.std(thickness)),
        }
        return result
    
    @staticmethod
    def extract_intensity_features(image):
        result = {
            'mean_intensity': float(np.mean(image)),
            'std_intensity': float(np.std(image)),
        }
        return result
    
    @staticmethod
    def extract_fluid_features(segmentation):
        result = {
            'has_fluid': float(segmentation['has_fluid']),
            'fluid_volume': float(segmentation['fluid_volume']),
        }
        return result
    
    @staticmethod
    def extract_entropy_features(image):
        hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()
        shannon_entropy = entropy(hist)
        return {'shannon_entropy': float(shannon_entropy)}
    
    @staticmethod
    def extract_all_features(image, disease_label=None):
        preprocessor = OCTPreprocessor()
        preprocessed = preprocessor.preprocess_pipeline(image)
        segmenter = OCTSegmenter()
        segmentation = segmenter.segment_retina(preprocessed)
        features = {}
        features.update(FeatureExtractor.extract_thickness_features(segmentation))
        features.update(FeatureExtractor.extract_intensity_features(preprocessed))
        features.update(FeatureExtractor.extract_fluid_features(segmentation))
        features.update(FeatureExtractor.extract_entropy_features(preprocessed))
        if disease_label is not None:
            features['disease_label'] = int(disease_label)
        return features
