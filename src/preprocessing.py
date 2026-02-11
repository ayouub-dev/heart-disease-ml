import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """Preprocessor for heart disease dataset."""
    
    def __init__(self, data_path, test_size=0.2, random_state=42):
        self.data_path = data_path
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        
    def load_data(self):
        """Load dataset from CSV file."""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.df.shape}")
        return self.df
    
    def handle_missing_values(self, X):
        """Handle missing values using median imputation."""
        if X.isnull().sum().sum() > 0:
            print(f"Found {X.isnull().sum().sum()} missing values. Imputing...")
            X_imputed = pd.DataFrame(
                self.imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            return X_imputed
        return X
    
    def engineer_features(self, df):
        """Create new features from existing ones."""
        df = df.copy()
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 100], 
                                  labels=[0, 1, 2, 3])
        
        # Cholesterol risk
        df['chol_risk'] = (df['chol'] > 240).astype(int)
        
        # Blood pressure risk
        df['bp_risk'] = (df['trestbps'] > 140).astype(int)
        
        # Maximum heart rate category
        df['thalach_category'] = pd.cut(df['thalach'], bins=[0, 100, 140, 180, 250],
                                         labels=[0, 1, 2, 3])
        
        # Convert categorical to numeric
        df['age_group'] = df['age_group'].astype(int)
        df['thalach_category'] = df['thalach_category'].astype(int)
        
        return df
    
    def scale_features(self, X_train, X_test):
        """Standardize features by removing mean and scaling to unit variance."""
        print("Scaling features...")
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self):
        """Complete preprocessing pipeline."""
        # Load data
        df = self.load_data()
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Separate features and target
        X = df.drop('target', axis=1)
        y = df['target']
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, 
            random_state=self.random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        print(f"Training set size: {X_train_scaled.shape[0]}")
        print(f"Test set size: {X_test_scaled.shape[0]}")
        print(f"Number of features: {X_train_scaled.shape[1]}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor('data/heart.csv')
    X_train, X_test, y_train, y_test = preprocessor.prepare_data()
    print("\nPreprocessing completed successfully!")
