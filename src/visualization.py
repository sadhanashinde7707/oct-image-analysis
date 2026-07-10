# src/visualization.py
# Visualization tools for OCT analysis

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class OCTVisualizer:
    """Visualization tools for OCT images and analysis results."""
    
    @staticmethod
    def plot_image_comparison(original, preprocessed, title="OCT Image Comparison"):
        """Plot original vs preprocessed image."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title('Original OCT Image')
        axes[0].axis('off')
        
        axes[1].imshow(preprocessed, cmap='gray')
        axes[1].set_title('Preprocessed OCT Image')
        axes[1].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_segmentation(image, segmentation, title="Retinal Layer Segmentation"):
        """Plot image with segmented boundaries."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.imshow(image, cmap='gray')
        
        # Plot ILM boundary
        ilm = segmentation['ilm']
        ax.plot(ilm, color='red', linewidth=2, label='ILM (Top)')
        
        # Plot RPE boundary
        rpe = segmentation['rpe']
        ax.plot(rpe, color='blue', linewidth=2, label='RPE (Bottom)')
        
        # Highlight fluid if present
        if segmentation['has_fluid']:
            fluid_mask = segmentation['fluid_mask']
            ax.contour(fluid_mask, colors='yellow', linewidths=1, levels=[0.5])
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (pixels)')
        ax.set_ylabel('Y Position (pixels)')
        ax.legend()
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(cm, disease_classes, model_name="Model"):
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        im = ax.imshow(cm, cmap='Blues')
        
        # Labels
        ax.set_xticks(np.arange(len(disease_classes)))
        ax.set_yticks(np.arange(len(disease_classes)))
        ax.set_xticklabels(disease_classes)
        ax.set_yticklabels(disease_classes)
        
        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations
        for i in range(len(disease_classes)):
            for j in range(len(disease_classes)):
                text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)
        
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_metrics_comparison(results, metric='f1'):
        """Plot metrics across models."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = [r['model_name'] for r in results]
        metric_values = [r[metric] for r in results]
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(model_names, metric_values, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(f'Model Comparison - {metric.upper()}', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def save_figure(fig, output_path, filename):
        """Save figure to file."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        plt.close(fig)
