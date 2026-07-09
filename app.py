# app.py
import numpy as np
from pathlib import Path
from src.datasets import OCTDataset
from src.features import FeatureExtractor

print("=" * 70)
print("OCT IMAGE ANALYSIS PIPELINE")
print("Testing Feature Extraction")
print("=" * 70)

data_path = Path("data/OCT2017 ")
dataset = OCTDataset(data_path)

print("\nLoading 8 sample images...")
images, labels, disease_names = dataset.load_all_images(split='train', max_images=8)

print(f"Loaded {len(images)} images")

feature_extractor = FeatureExtractor()
all_features = []

print("\n" + "=" * 70)
print("FEATURE EXTRACTION RESULTS")
print("=" * 70)

for idx in range(len(images)):
    print(f"\n--- Image {idx+1}: {disease_names[idx]} ---")
    features = feature_extractor.extract_all_features(images[idx], disease_label=labels[idx])
    all_features.append(features)
    
    print(f"  THICKNESS:")
    print(f"    Mean: {features['mean_thickness']:.1f} pixels")
    print(f"    Std: {features['std_thickness']:.1f} pixels")
    print(f"  INTENSITY:")
    print(f"    Mean: {features['mean_intensity']:.1f}")
    print(f"    Std: {features['std_intensity']:.1f}")
    print(f"  FLUID:")
    print(f"    Has Fluid: {bool(features['has_fluid'])}")
    print(f"    Volume: {features['fluid_volume']:.0f} pixels")
    print(f"  ENTROPY:")
    print(f"    Shannon: {features['shannon_entropy']:.3f}")

print("\n" + "=" * 70)
print(f"✓ Extracted {len(all_features)} images")
print(f"✓ Features per image: {len(all_features[0])}")
print(f"✓ Ready for Machine Learning!")
print("=" * 70 + "\n")
