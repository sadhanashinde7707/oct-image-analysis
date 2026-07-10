# src/classification.py
# Machine learning classification models

import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

class OCTClassifier:
    """Machine learning classifier for OCT disease classification."""
    
    def __init__(self):
        """Initialize classifiers."""
        self.scaler = StandardScaler()
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.svm_model = SVC(kernel='rbf', random_state=42, probability=True)
        self.models = {
            'Logistic Regression': self.lr_model,
            'Random Forest': self.rf_model,
            'SVM': self.svm_model
        }
    
    def prepare_data(self, features_list, labels):
        """
        Prepare features and labels for training.
        
        Args:
            features_list: List of feature dictionaries
            labels: Array of disease labels
            
        Returns:
            X: Feature matrix (scaled)
            y: Label vector
        """
        # Extract feature values (excluding disease_label)
        X = []
        for features in features_list:
            feature_vector = [v for k, v in features.items() if k != 'disease_label']
            X.append(feature_vector)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train_all_models(self, X_train, y_train):
        """Train all classifiers."""
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            print(f"  ✓ {model_name} trained")
    
    def evaluate_model(self, model_name, model, X_test, y_test):
        """Evaluate single model."""
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # ROC-AUC for multiclass
        try:
            if len(np.unique(y_test)) > 2:
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr', average='weighted')
            else:
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except:
            roc_auc = 0.0
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all classifiers."""
        results = []
        for model_name, model in self.models.items():
            result = self.evaluate_model(model_name, model, X_test, y_test)
            results.append(result)
        return results
    
    def print_results(self, results, disease_classes):
        """Print evaluation results."""
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE")
        print("=" * 70)
        
        for result in results:
            print(f"\n{result['model_name']}:")
            print(f"  Accuracy:  {result['accuracy']:.3f}")
            print(f"  Precision: {result['precision']:.3f}")
            print(f"  Recall:    {result['recall']:.3f}")
            print(f"  F1-Score:  {result['f1']:.3f}")
            print(f"  ROC-AUC:   {result['roc_auc']:.3f}")
        
        # Find best model
        best_model = max(results, key=lambda x: x['f1'])
        print(f"\n✓ Best Model: {best_model['model_name']} (F1: {best_model['f1']:.3f})")
