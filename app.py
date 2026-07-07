# app.py
# OCT Image Analysis Pipeline - Test Preprocessing

import numpy as np
import cv2
from pathlib import Path
from src.datasets import OCTDataset
from src.preprocessing import OCTPreprocessor

print("=" * 70)
print("OCT IMAGE ANALYSIS PIPELINE")
print("Testing Image Preprocessing")
print("=" * 70)

# Initialize dataset
data_path = Path("data/OCT2017 ")
dataset = OCTDataset(data_path)

print("\n✓ Loading 5 sample NORMAL images...")
images, labels, disease_names = dataset.load_all_images(split='train', max_images=5)

print(f"✓ Loaded {len(images)} images")
print(f"✓ Image shape: {images.shape}")

# Test preprocessing on first image
print("\n" + "=" * 70)
print("PREPROCESSING PIPELINE TEST")
print("=" * 70)

sample_image = images[0]
print(f"\nOriginal Image Stats:")
print(f"  Shape: {sample_image.shape}")
print(f"  Data type: {sample_image.dtype}")
print(f"  Min: {sample_image.min()}, Max: {sample_image.max()}")
print(f"  Mean: {sample_image.mean():.2f}, Std: {sample_image.std():.2f}")

# Apply preprocessing
preprocessor = OCTPreprocessor()
processed_image = preprocessor.preprocess_pipeline(sample_image, apply_clahe=True, denoise_method='bilateral')

print(f"\nPreprocessed Image Stats:")
print(f"  Shape: {processed_image.shape}")
print(f"  Data type: {processed_image.dtype}")
print(f"  Min: {processed_image.min()}, Max: {processed_image.max()}")
print(f"  Mean: {processed_image.mean():.2f}, Std: {processed_image.std():.2f}")

# Calculate improvement metrics
contrast_improvement = processed_image.std() / sample_image.std() if sample_image.std() > 0 else 1.0
print(f"\nContrast Improvement: {contrast_improvement:.2f}x")

print("\n" + "=" * 70)
print("✓ Preprocessing pipeline working successfully!")
print("✓ Images ready for segmentation and classification")
print("=" * 70 + "\n")
