# app.py
import numpy as np
import cv2
from pathlib import Path
from src.datasets import OCTDataset

print("=" * 70)
print("OCT IMAGE ANALYSIS PIPELINE")
print("Retinal Layer Segmentation & Disease Classification")
print("=" * 70)

data_path = Path("data/OCT2017 ")
dataset = OCTDataset(data_path)

print("\n✓ Dataset initialized successfully")
print(f"✓ Disease classes: {dataset.disease_classes}")

print("\n" + "=" * 70)
print("LOADING SAMPLE IMAGES FROM TRAINING SET")
print("=" * 70)

images, labels, disease_names = dataset.load_all_images(split='train', max_images=10)

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)
print(f"✓ Total images loaded: {len(images)}")
print(f"✓ Image shape: {images.shape}")
print(f"✓ Min: {images.min()}, Max: {images.max()}")

unique, counts = np.unique(labels, return_counts=True)
print("\nDISEASE DISTRIBUTION:")
for disease_idx, count in zip(unique, counts):
    disease_name = dataset.disease_classes[disease_idx]
    print(f"  {disease_name}: {count} images")

print("\n✓ Ready for preprocessing!\n")
