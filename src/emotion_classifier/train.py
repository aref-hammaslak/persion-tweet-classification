import pandas as pd
from pathlib import Path
from typing import List, Dict
from .preprocessing import TextPreprocessor
from .model import EmotionClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_dataset(dataset_dir: Path, categories: List[str]) -> pd.DataFrame:
    """Load and combine datasets from different categories."""
    dataset_df = pd.DataFrame()
    
    for category in categories:
        print(f"Processing {category} category")
        file = dataset_dir / 'raw' / f"{category}.csv"
        df = pd.read_csv(file)
        df['label'] = category
        dataset_df = pd.concat([dataset_df, df], axis=0)
    
    return dataset_df

def train_emotion_classifier(
    dataset_dir: Path,
    model_dir: Path,
    categories: List[str] = ['anger', 'fear', 'joy', 'sad', 'disgust', 'surprise']
) -> Dict:
    """Train the emotion classifier and return evaluation metrics."""
    # Initialize components
    preprocessor = TextPreprocessor()
    classifier = EmotionClassifier()
    
    # Load and preprocess data
    dataset_df = load_dataset(dataset_dir, categories)
    dataset_df['processed_text'] = dataset_df['tweet'].apply(preprocessor.process_text)
    
    # Print dataset statistics
    print("\nDataset distribution:")
    print(dataset_df['label'].value_counts())
    
    # Prepare features and train model
    X = classifier.prepare_features(dataset_df['processed_text'])
    y = dataset_df['label'].values
    
    X_train, X_test, y_train, y_test = classifier.train(X, y)
    classifier.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Save model
    classifier.save(model_dir)
    
    return {
        'accuracy': accuracy,
        'classification_report': report
    } 