# app.py
# OCT Image Analysis Pipeline - Test Segmentation

import numpy as np
import cv2
from pathlib import Path
from src.datasets import OCTDataset
from src.preprocessing import OCTPreprocessor
from src.segmentation import OCTSegmenter

print("=" * 70)
print("OCT IMAGE ANALYSIS PIPELINE")
print("Testing Retinal Layer Segmentation")
print("=" * 70)

# Initialize dataset
data_path = Path("data/OCT2017 ")
dataset = OCTDataset(data_path)

print("\n✓ Loading sample images from each disease category...")
images, labels, disease_names = dataset.load_all_images(split='train', max_images=4)

print(f"✓ Loaded {len(images)} images")

# Test segmentation on each image
preprocessor = OCTPreprocessor()
segmenter = OCTSegmenter()

print("\n" + "=" * 70)
print("SEGMENTATION RESULTS")
print("=" * 70)

for idx in range(min(4, len(images))):
    print(f"\n--- Image {idx+1}: {disease_names[idx]} ---")
    
    # Get original image
    original = images[idx]
    
    # Preprocess
    preprocessed = preprocessor.preprocess_pipeline(original)
    
    # Segment
    segmentation = segmenter.segment_retina(preprocessed)
    
    # Print results
    print(f"  Mean Retinal Thickness: {segmentation['mean_thickness']:.1f} pixels")
    print(f"  Fluid Detected: {segmentation['has_fluid']}")
    print(f"  Fluid Volume: {segmentation['fluid_volume']} pixels")
    print(f"  ILM Range: {segmentation['ilm'].min():.0f} - {segmentation['ilm'].max():.0f}")
    print(f"  RPE Range: {segmentation['rpe'].min():.0f} - {segmentation['rpe'].max():.0f}")

print("\n" + "=" * 70)
print("✓ Segmentation pipeline working successfully!")
print("✓ Retinal layers and abnormalities identified")
print("=" * 70 + "\n")
