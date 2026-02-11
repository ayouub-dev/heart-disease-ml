import argparse
import os
from .preprocessing import DataPreprocessor
from .model import HeartDiseaseModel
import pandas as pd


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train heart disease prediction model'
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        default='data/heart.csv',
        help='Path to the dataset CSV file'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        default='random_forest',
        choices=['logistic_regression', 'random_forest', 
                 'gradient_boosting', 'svm', 'xgboost'],
        help='Type of model to train'
    )
    
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Proportion of dataset to use for testing'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    parser.add_argument(
        '--compare_models',
        action='store_true',
        help='Train and compare multiple models'
    )
    
    parser.add_argument(
        '--cv_folds',
        type=int,
        default=5,
        help='Number of cross-validation folds'
    )
    
    return parser.parse_args()


def train_single_model(args, X_train, X_test, y_train, y_test):
    """Train a single model."""
    print(f"\n{'='*60}")
    print(f"Training {args.model_type} model")
    print(f"{'='*60}\n")
    
    # Initialize and train model
    model = HeartDiseaseModel(
        model_type=args.model_type,
        random_state=args.random_state
    )
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics = model.evaluate(X_test, y_test)
    
    # Cross-validation
    print("\nPerforming cross-validation...")
    cv_scores = model.cross_validate(X_train, y_train, cv=args.cv_folds)
    
    # Save model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f'{args.model_type}_model.pkl')
    model.save_model(model_path)
    
    return model, metrics


def compare_models(args, X_train, X_test, y_train, y_test):
    """Train and compare multiple models."""
    model_types = ['logistic_regression', 'random_forest', 
                   'gradient_boosting', 'xgboost', 'svm']
    
    results = []
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type}")
        print(f"{'='*60}\n")
        
        model = HeartDiseaseModel(
            model_type=model_type,
            random_state=args.random_state
        )
        model.train(X_train, y_train, verbose=False)
        metrics = model.evaluate(X_test, y_test, verbose=True)
        
        results.append({
            'model': model_type,
            **metrics
        })
        
        # Save model
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f'{model_type}_model.pkl')
        model.save_model(model_path)
    
    # Create comparison DataFrame
    results_df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}\n")
    print(results_df.to_string(index=False))
    
    # Save results
    results_path = os.path.join(args.output_dir, 'model_comparison.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")
    
    # Find best model
    best_model = results_df.loc[results_df['accuracy'].idxmax(), 'model']
    best_accuracy = results_df['accuracy'].max()
    print(f"\nBest model: {best_model} (Accuracy: {best_accuracy:.4f})")
    
    return results_df


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("="*60)
    print("HEART DISEASE PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Preprocess data
    print("\n1. PREPROCESSING DATA")
    print("-"*60)
    preprocessor = DataPreprocessor(
        data_path=args.data_path,
        test_size=args.test_size,
        random_state=args.random_state
    )
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    
    # Train model(s)
    print("\n2. TRAINING MODEL(S)")
    print("-"*60)
    
    if args.compare_models:
        compare_models(args, X_train, X_test, y_train, y_test)
    else:
        train_single_model(args, X_train, X_test, y_train, y_test)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()
