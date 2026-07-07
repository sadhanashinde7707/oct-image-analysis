# src/datasets.py
import os
import cv2
import numpy as np
from pathlib import Path

class OCTDataset:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.disease_classes = ['NORMAL', 'CNV', 'DME', 'DRUSEN']
        self.class_labels = {disease: idx for idx, disease in enumerate(self.disease_classes)}
        
    def load_image(self, image_path, size=(512, 512)):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        img_resized = cv2.resize(img, size)
        return img_resized
    
    def load_all_images(self, split='train', max_images=100):
        images = []
        labels = []
        disease_names = []
        split_path = self.data_path / split
        count = 0
        for disease in self.disease_classes:
            disease_path = split_path / disease
            if not disease_path.exists():
                print(f"Warning: {disease_path} not found")
                continue
            image_files = list(disease_path.glob('*.jpeg'))
            print(f"\nLoading {disease} images ({len(image_files)} found)...")
            for image_file in image_files[:max_images]:
                try:
                    img = self.load_image(image_file)
                    images.append(img)
                    labels.append(self.class_labels[disease])
                    disease_names.append(disease)
                    count += 1
                    if count % 100 == 0:
                        print(f"  Loaded {count} images...")
                except Exception as e:
                    print(f"  Error loading {image_file}: {e}")
        images = np.array(images)
        labels = np.array(labels)
        print(f"\nLoaded {len(images)} total images")
        print(f"Image shape: {images.shape}")
        return images, labels, disease_names
