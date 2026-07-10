# app.py
# Complete OCT Analysis Pipeline with Visualization

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.datasets import OCTDataset
from src.features import FeatureExtractor
from src.preprocessing import OCTPreprocessor
from src.segmentation import OCTSegmenter
from src.classification import OCTClassifier
from src.visualization import OCTVisualizer

print("=" * 70)
print("OCT IMAGE ANALYSIS PIPELINE - COMPLETE")
print("=" * 70)

# 1. Load Dataset
print("\n[1/6] Loading dataset...")
data_path = Path("data/OCT2017 ")
dataset = OCTDataset(data_path)
images, labels, disease_names = dataset.load_all_images(split='train', max_images=40)
print(f"✓ Loaded {len(images)} images")

# 2. Extract Features
print("\n[2/6] Extracting features...")
feature_extractor = FeatureExtractor()
all_features = []
for idx, (image, label) in enumerate(zip(images, labels)):
    if (idx + 1) % 10 == 0:
        print(f"  Processed {idx + 1}/{len(images)}")
    features = feature_extractor.extract_all_features(image, disease_label=label)
    all_features.append(features)
print(f"✓ Extracted features for {len(all_features)} images")

# 3. Train Models
print("\n[3/6] Training ML models...")
classifier = OCTClassifier()
X, y = classifier.prepare_data(all_features, labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
classifier.train_all_models(X_train, y_train)
print(f"✓ Trained 3 models on {len(X_train)} images")

# 4. Evaluate Models
print("\n[4/6] Evaluating models...")
disease_classes = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
results = classifier.evaluate_all_models(X_test, y_test)
classifier.print_results(results, disease_classes)
print(f"✓ Evaluated {len(X_test)} test images")

# 5. Create Visualizations
print("\n[5/6] Creating visualizations...")
visualizer = OCTVisualizer()
figures_dir = Path("figures")

# Plot confusion matrices
for result in results:
    cm = result['confusion_matrix']
    fig = visualizer.plot_confusion_matrix(cm, disease_classes, result['model_name'])
    visualizer.save_figure(fig, figures_dir, f"confusion_matrix_{result['model_name'].replace(' ', '_').lower()}.png")

# Plot metrics comparison
fig = visualizer.plot_metrics_comparison(results, metric='f1')
visualizer.save_figure(fig, figures_dir, "metrics_f1_comparison.png")

fig = visualizer.plot_metrics_comparison(results, metric='accuracy')
visualizer.save_figure(fig, figures_dir, "metrics_accuracy_comparison.png")

# Plot sample segmentations
print("  Creating sample segmentations...")
preprocessor = OCTPreprocessor()
segmenter = OCTSegmenter()

for idx in range(min(3, len(images))):
    preprocessed = preprocessor.preprocess_pipeline(images[idx])
    segmentation = segmenter.segment_retina(preprocessed)
    
    fig = visualizer.plot_image_comparison(images[idx], preprocessed, f"Image {idx+1}: {disease_names[idx]}")
    visualizer.save_figure(fig, figures_dir, f"preprocessing_sample_{idx+1}.png")
    
    fig = visualizer.plot_segmentation(preprocessed, segmentation, f"Segmentation {idx+1}: {disease_names[idx]}")
    visualizer.save_figure(fig, figures_dir, f"segmentation_sample_{idx+1}.png")

print(f"✓ Created visualizations in {figures_dir}/")

# 6. Summary
print("\n[6/6] Pipeline Summary")
print("=" * 70)
print(f"✓ Dataset: {len(images)} OCT images (4 disease categories)")
print(f"✓ Features: {X.shape[1]} clinical features per image")
print(f"✓ Models: 3 ML classifiers trained")
print(f"✓ Best Model: {max(results, key=lambda x: x['f1'])['model_name']}")
print(f"✓ Best F1-Score: {max(results, key=lambda x: x['f1'])['f1']:.3f}")
print(f"✓ Visualizations: Saved to figures/")
print("=" * 70)
print("\n✓ OCT Analysis Pipeline Complete!")
print("✓ Ready for deployment and clinical validation\n")
