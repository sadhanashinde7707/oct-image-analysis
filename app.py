# app.py
# OCT Image Analysis Pipeline - Train & Evaluate ML Models

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.datasets import OCTDataset
from src.features import FeatureExtractor
from src.classification import OCTClassifier

print("=" * 70)
print("OCT IMAGE ANALYSIS PIPELINE")
print("Machine Learning Classification")
print("=" * 70)

# Load dataset
data_path = Path("data/OCT2017 ")
dataset = OCTDataset(data_path)

print("\n✓ Loading 40 sample images (10 per disease)...")
images, labels, disease_names = dataset.load_all_images(split='train', max_images=40)

print(f"✓ Loaded {len(images)} images")
print(f"  NORMAL: {np.sum(labels == 0)}")
print(f"  CNV:    {np.sum(labels == 1)}")
print(f"  DME:    {np.sum(labels == 2)}")
print(f"  DRUSEN: {np.sum(labels == 3)}")

# Extract features
print("\n" + "=" * 70)
print("EXTRACTING FEATURES")
print("=" * 70)

feature_extractor = FeatureExtractor()
all_features = []

for idx, (image, label) in enumerate(zip(images, labels)):
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(images)} images...")
    features = feature_extractor.extract_all_features(image, disease_label=label)
    all_features.append(features)

print(f"✓ Extracted features for {len(all_features)} images")

# Prepare data
print("\n" + "=" * 70)
print("TRAINING ML MODELS")
print("=" * 70)

classifier = OCTClassifier()
X, y = classifier.prepare_data(all_features, labels)

print(f"\n✓ Features shape: {X.shape}")
print(f"✓ Labels shape: {y.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"✓ Training set: {X_train.shape[0]} images")
print(f"✓ Test set: {X_test.shape[0]} images")

# Train models
print(f"\nTraining models...")
classifier.train_all_models(X_train, y_train)

# Evaluate
print("\n" + "=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

disease_classes = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
results = classifier.evaluate_all_models(X_test, y_test)
classifier.print_results(results, disease_classes)

print("\n" + "=" * 70)
print("✓ Pipeline Complete!")
print("✓ All models trained and evaluated successfully")
print("=" * 70 + "\n")
