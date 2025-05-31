import re
import hazm
from hazm import Stemmer, Normalizer, Lemmatizer, word_tokenize
from typing import List, Callable

class TextPreprocessor:
    def __init__(self):
        self.stemmer = Stemmer()
        self.stopwords = hazm.stopwords_list()
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer()
    
    def normalize_text(self, text: str) -> str:
        """Normalize the input text."""
        text = self.normalizer.normalize(text)
        text = text.lower()
        text = re.sub(r'[^آ-ی\s]', '', text)
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize the input text."""
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        tokens = [self.stemmer.stem(token) for token in tokens]
        tokens = [token for token in tokens if token not in self.stopwords]
        tokens = [token for token in tokens if len(token) > 2]
        return tokens
    
    def process_text(self, text: str) -> str:
        """Process text through the complete pipeline."""
        text = self.normalize_text(text)
        tokens = self.tokenize(text)
        return ' '.join(tokens) 