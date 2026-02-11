import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import joblib


class HeartDiseaseModel:
    """Heart disease prediction model with multiple classifier options."""
    
    def __init__(self, model_type='random_forest', random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._initialize_model()
        self.is_trained = False
        
    def _initialize_model(self):
        """Initialize the specified model type."""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, 
                random_state=self.random_state, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=5, random_state=self.random_state
            ),
            'svm': SVC(
                kernel='rbf', probability=True,
                random_state=self.random_state
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100, learning_rate=0.1,
                max_depth=5, random_state=self.random_state,
                eval_metric='logloss'
            )
        }
        
        if self.model_type not in models:
            raise ValueError(f"Model type {self.model_type} not supported.")
        
        return models[self.model_type]
    
    def train(self, X_train, y_train, verbose=True):
        """Train the model on training data."""
        if verbose:
            print(f"Training {self.model_type} model...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        if verbose:
            print("Training completed!")
        
        return self
    
    def predict(self, X):
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test, verbose=True):
        """Evaluate model performance on test data."""
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        if verbose:
            print(f"\n{'='*50}")
            print(f"Model: {self.model_type}")
            print(f"{'='*50}")
            for metric, value in metrics.items():
                print(f"{metric.capitalize()}: {value:.4f}")
            
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))
        
        return metrics
    
    def cross_validate(self, X, y, cv=5, verbose=True):
        """Perform cross-validation."""
        scores = cross_val_score(
            self.model, X, y, cv=cv, 
            scoring='accuracy', n_jobs=-1
        )
        
        if verbose:
            print(f"\nCross-validation scores: {scores}")
            print(f"Mean CV accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, filepath):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model.")
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath, model_type):
        """Load a trained model from disk."""
        model_instance = HeartDiseaseModel(model_type=model_type)
        model_instance.model = joblib.load(filepath)
        model_instance.is_trained = True
        print(f"Model loaded from {filepath}")
        return model_instance


if __name__ == "__main__":
    # Test model initialization
    model = HeartDiseaseModel(model_type='random_forest')
    print(f"Model initialized: {model.model_type}")
