from pathlib import Path
from emotion_classifier import train_emotion_classifier

def main():
    # Set up directories
    dataset_dir = (Path().absolute() / '../dataset').resolve()
    model_dir = (Path().absolute() / '../models').resolve()
    
    # Train the model
    metrics = train_emotion_classifier(dataset_dir, model_dir)
    
    # Print results
    print(f"\nModel Accuracy: {metrics['accuracy']:.2f}")
    print("\nDetailed Classification Report:")
    print(metrics['classification_report'])

if __name__ == "__main__":
    main() 