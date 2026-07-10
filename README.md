# OCT Image Analysis Pipeline

**Retinal Layer Segmentation and Disease Classification Using Python**

![Python](https://img.shields.io/badge/Python-3.14-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## Overview

A complete machine learning pipeline for analyzing Optical Coherence Tomography (OCT) images to classify retinal diseases. This project demonstrates end-to-end medical imaging analysis: **preprocessing → segmentation → feature extraction → classification**.

**Clinical Applications:**
- Automated screening for diabetic macular edema (DME)
- Detection of choroidal neovascularization (CNV)
- Age-related macular degeneration (AMD) assessment
- Support for ophthalmology decision-making

## Dataset

- **OCT2017 Dataset** (Kaggle): 83,484 high-resolution OCT images
- **4 Disease Categories:** NORMAL, CNV, DME, DRUSEN
- **Train/Val/Test Split:** Complete with validated partitions
- **Image Resolution:** 512×512 pixels (grayscale)

## Pipeline Architecture
## Project Structure
## Installation

### Prerequisites
- Python 3.10+
- Git
- Virtual environment (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/sadhanashinde7707/oct-image-analysis.git
cd oct-image-analysis

# Create virtual environment
python -m venv .venv
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Run Complete Pipeline

```bash
python app.py
```

Output:
- Loads 40 sample images across 4 disease categories
- Preprocesses and segments retinal layers
- Extracts 15+ clinical features per image
- Trains 3 ML models
- Evaluates performance (accuracy, F1, ROC-AUC)
- Generates confusion matrices & comparison plots

### 2. Interactive Notebook

```bash
jupyter notebook notebooks/walkthrough.ipynb
```

Step-by-step demonstration with visualizations and explanations.

### 3. Custom Analysis

```python
from src.datasets import OCTDataset
from src.features import FeatureExtractor
from src.classification import OCTClassifier

# Load images
dataset = OCTDataset("data/OCT2017 ")
images, labels, _ = dataset.load_all_images(split='train', max_images=50)

# Extract features
extractor = FeatureExtractor()
features = [extractor.extract_all_features(img) for img in images]

# Classify
clf = OCTClassifier()
X, y = clf.prepare_data(features, labels)
# ... train and evaluate
```

## Key Components

### 1. Data Loading (`src/datasets.py`)
- Loads JPEG images from organized directories
- Handles 4 disease categories
- Resizes to standard 512×512 pixels
- Provides train/val/test splits

### 2. Preprocessing (`src/preprocessing.py`)
**Why:** OCT images have characteristic speckle noise and low contrast in some regions

**Methods:**
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Enhances retinal layer contrast while limiting noise amplification
- **Bilateral Filtering**: Edge-preserving denoising (preserves retinal boundaries)
- **Normalization**: Scales pixel values to [0, 1] range

### 3. Segmentation (`src/segmentation.py`)
**What:** Identifies anatomical retinal layers and pathology

**Outputs:**
- **ILM (Internal Limiting Membrane)**: Top boundary of neurosensory retina
- **RPE (Retinal Pigment Epithelium)**: Outer boundary
- **Fluid Detection**: Abnormal dark regions indicating disease
- **Retinal Thickness**: Clinical parameter for disease monitoring

### 4. Feature Extraction (`src/features.py`)
**Purpose:** Convert images to numerical features for ML

**Extracted Features (15 total):**
- Thickness: mean, std, max, min
- Intensity: mean, std, median, min, max
- Fluid: presence, volume, percentage
- Entropy: Shannon entropy (image disorder)

### 5. Classification (`src/classification.py`)
**Models Implemented:**

| Model | Strength | Best For |
|-------|----------|----------|
| **Logistic Regression** | Interpretable, fast | Baseline, linear patterns |
| **Random Forest** | Handles non-linearity | Feature interactions |
| **SVM** | High-dimensional data | Robust classification |

**Metrics Computed:**
- Accuracy, Precision, Recall, F1-score
- ROC-AUC (multi-class)
- Confusion matrices

### 6. Visualization (`src/visualization.py`)
- Original vs. preprocessed images
- Segmentation overlays with boundaries
- Confusion matrices per model
- Metrics comparison charts

## Results

### Model Performance (40-image sample)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| Random Forest | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |
| SVM | 0.XXX | 0.XXX | 0.XXX | 0.XXX | 0.XXX |

*Note: Performance scales with dataset size (40 vs. 83K images)*

### Outputs

- **Confusion matrices**: `figures/confusion_matrix_*.png`
- **Metrics comparison**: `figures/metrics_*.png`
- **Segmentation examples**: `figures/segmentation_sample_*.png`
- **Preprocessing examples**: `figures/preprocessing_sample_*.png`

## Clinical Significance

### Why This Matters

1. **Screening Efficiency**: AI can pre-screen thousands of images, flagging suspicious cases for specialist review
2. **Early Detection**: Automated systems can identify subtle disease markers missed by manual review
3. **Consistency**: Removes observer bias in disease detection
4. **Scalability**: Enables population-wide screening programs

### Validation & Deployment

For clinical use, this pipeline requires:
- ✓ Validation on 1000+ independent images
- ✓ Comparison with ophthalmologist gold-standard
- ✓ Sensitivity/specificity analysis per disease
- ✓ FDA/CE approval process
- ✓ Integration with clinical PACS systems

## Technologies & Libraries

| Layer | Technologies |
|-------|--------------|
| **Data Processing** | NumPy, Pandas, OpenCV, SciPy |
| **ML Frameworks** | scikit-learn (models, metrics, preprocessing) |
| **Visualization** | Matplotlib |
| **Development** | Python 3.14, Git, Jupyter |
| **Data Source** | Kaggle OCT2017 Dataset |

## Future Enhancements

### Phase 2: Deep Learning
- Convolutional Neural Networks (CNN) for automatic feature learning
- Transfer learning (ResNet, VGG pre-trained)
- End-to-end deep segmentation U-Net

### Phase 3: Clinical Integration
- Real-time prediction API (Flask/FastAPI)
- Integration with ophthalmology EHR systems
- Mobile app for field screening

### Phase 4: Advanced Analytics
- Attention mechanisms to explain predictions
- Ensemble methods combining models
- Uncertainty quantification

## Installation from Source

```bash
# Clone and setup
git clone https://github.com/sadhanashinde7707/oct-image-analysis.git
cd oct-image-analysis
python -m venv .venv
source .venv/Scripts/activate
pip install -r requirements.txt

# Download dataset (requires Kaggle API)
kaggle datasets download -d paultimothymooney/kermany2018 -p data/
unzip data/kermany2018.zip -d data/

# Run pipeline
python app.py
```

## References

**Dataset:**
- Kermany, D. S., et al. (2018). Labeled Optical Coherence Tomography (OCT) and Chest X-ray Images for Automated Diagnosis. Mendeley Data. https://data.mendeley.com/datasets/rscbjbr9sj

**Medical Background:**
- Ting, D. S., et al. (2019). AI for medical imaging goes deep. *Nature Medicine*, 25(1), 13.
- Topol, E. J. (2019). High-performance medicine. *The Lancet*, 392(10157), 1541-1546.

## Author

**Sadhana Shinde**
- Master's in Digital Health, Deggendorf Institute of Technology, Germany
- Background: Clinical Optometry, Healthcare Operations
- Focus: Medical Imaging, Digital Health, MedTech
- GitHub: [@sadhanashinde7707](https://github.com/sadhanashinde7707)

## License

MIT License - See LICENSE file for details

## Contact & Support

- **Issues**: GitHub Issues for bug reports
- **Email**: sadhana@example.com
- **Portfolio**: [GitHub Profile](https://github.com/sadhanashinde7707)

---

**Last Updated:** July 2026  
**Status:** Active Development
