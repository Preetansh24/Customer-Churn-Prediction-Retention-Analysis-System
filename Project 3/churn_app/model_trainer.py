"""
Customer Churn Prediction Model Trainer
Trains and saves the churn prediction model from telco_churn.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def train_and_save_model():
    print("=" * 60)
    print("CUSTOMER CHURN MODEL TRAINER")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv('../telco_churn.csv')
    print(f"   Loaded {len(df)} records")
    
    # Data cleaning
    print("\n2. Cleaning data...")
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    df['ChurnBinary'] = (df['Churn'] == 'Yes').astype(int)
    df = df.drop('Churn', axis=1)
    print("   Data cleaned successfully")
    
    # Define features
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 
                       'PhoneService', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                       'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
    
    # Convert SeniorCitizen to string for consistency
    df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
    
    X = df[numeric_cols + categorical_cols]
    y = df['ChurnBinary']
    
    # Train-test split
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Create preprocessor
    print("\n4. Creating preprocessor...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ]
    )
    
    # Create and train model pipeline
    print("\n5. Training Random Forest model...")
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    print("   Model trained successfully")
    
    # Evaluate
    print("\n6. Evaluating model...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"   Training accuracy: {train_score:.4f}")
    print(f"   Test accuracy: {test_score:.4f}")
    
    # Get feature names
    print("\n7. Extracting feature names...")
    preprocessor.fit(X_train)
    cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    all_features = list(numeric_cols) + list(cat_features)
    print(f"   Total features: {len(all_features)}")
    
    # Save model and artifacts
    print("\n8. Saving model and artifacts...")
    os.makedirs('model', exist_ok=True)
    
    joblib.dump(model, 'model/churn_model.pkl')
    print("   Saved: model/churn_model.pkl")
    
    joblib.dump(all_features, 'model/feature_names.pkl')
    print("   Saved: model/feature_names.pkl")
    
    # Save feature info for the app
    feature_info = {
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols
    }
    joblib.dump(feature_info, 'model/feature_info.pkl')
    print("   Saved: model/feature_info.pkl")
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETE!")
    print("=" * 60)
    
    return model, all_features

if __name__ == "__main__":
    train_and_save_model()
