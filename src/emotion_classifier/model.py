from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import joblib
from pathlib import Path
from typing import Tuple, Any, List
import numpy as np
from scipy.sparse import csr_matrix

class EmotionClassifier:
    def __init__(self, 
                 max_features: int = 10000,
                 min_df: int = 5,
                 max_df: float = 0.95,
                 ngram_range: Tuple[int, int] = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range
        )
        self.model = MultinomialNB()
        self.oversampler = RandomOverSampler(random_state=0)
    
    def prepare_features(self, texts: List[str]) -> csr_matrix:
        """Transform texts to TF-IDF features."""
        return self.vectorizer.fit_transform(texts)
    
    def balance_dataset(self, X: csr_matrix, y: np.ndarray) -> Tuple[csr_matrix, np.ndarray]:
        """Balance the dataset using random oversampling."""
        return self.oversampler.fit_resample(X, y)
    
    def train(self, X: csr_matrix, y: np.ndarray, test_size: float = 0.2) -> Tuple[Any, Any, Any, Any]:
        """Train the model with the given data."""
        X_resampled, y_resampled = self.balance_dataset(X, y)
        return train_test_split(X_resampled, y_resampled, test_size=test_size, random_state=0)
    
    def fit(self, X_train: csr_matrix, y_train: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X_train, y_train)
    
    def predict(self, X: csr_matrix) -> np.ndarray:
        """Make predictions on new data."""
        return self.model.predict(X)
    
    def save(self, model_dir: Path) -> None:
        """Save the model and vectorizer."""
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_dir / 'emotion_classifier.joblib')
        joblib.dump(self.vectorizer, model_dir / 'tfidf_vectorizer.joblib')
    
    @classmethod
    def load(cls, model_dir: Path) -> 'EmotionClassifier':
        """Load a saved model."""
        classifier = cls()
        classifier.model = joblib.load(model_dir / 'emotion_classifier.joblib')
        classifier.vectorizer = joblib.load(model_dir / 'tfidf_vectorizer.joblib')
        return classifier 