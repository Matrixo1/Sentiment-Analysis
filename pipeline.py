import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc,
)

from data_loader import load_imdb_dataset
from preprocess import clean_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.exists(os.path.join(current_dir, "artifacts")):
    ARTIFACTS_DIR = os.path.join(current_dir, "artifacts")
else:
    ARTIFACTS_DIR = os.path.abspath(os.path.join(current_dir, "../artifacts"))
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def train_and_evaluate(sample_limit=None):
    """
    Runs the complete sentiment pipeline:
    1. Load IMDB movie review dataset.
    2. Preprocess reviews (clean text, tokenize, lemmatize).
    3. Split train/test (80/20).
    4. Extract TF-IDF & Bag of Words features.
    5. Train models (Logistic Regression, Linear SVM, Naive Bayes).
    6. Evaluate and save metrics/plots.
    7. Save the best model & vectorizer.
    """
    logger.info("Starting pipeline...")
    
    # 1. Load data
    train_df, test_df = load_imdb_dataset()
    
    # Combine datasets to allow custom splitting if needed or maintain the standard split
    # Since IMDB dataset from HF has 25k train / 25k test (50/50 split), we can combine and split 80/20
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Apply sample limit for faster training/validation if requested
    if sample_limit:
        logger.info(f"Limiting dataset to {sample_limit} samples for speed.")
        df = df.sample(n=sample_limit, random_state=42).reset_index(drop=True)
        
    logger.info("Preprocessing reviews...")
    # Preprocess the reviews
    df["clean_text"] = df["text"].apply(clean_text)
    
    # Remove empty cleaned texts if any
    df = df[df["clean_text"] != ""].reset_index(drop=True)
    
    logger.info("Splitting dataset into train/test (80/20)...")
    # 2. Split data (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    logger.info(f"Train size: {len(X_train_raw)}, Test size: {len(X_test_raw)}")
    
    # Save test set raw data for reference or sample verification
    test_data_path = os.path.join(ARTIFACTS_DIR, "test_samples.csv")
    pd.DataFrame({"text": X_test_raw, "label": y_test}).to_csv(test_data_path, index=False)
    
    # 3. Feature Extraction
    logger.info("Extracting TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_raw)
    
    logger.info("Extracting CountVectorizer (Bag of Words) features...")
    bow_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_bow = bow_vectorizer.fit_transform(X_train_raw)
    X_test_bow = bow_vectorizer.transform(X_test_raw)
    
    # 4. Models to train
    models_config = {
        "Logistic Regression (TF-IDF)": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "train_features": X_train_tfidf,
            "test_features": X_test_tfidf,
            "prob_support": True
        },
        "Linear SVM (TF-IDF)": {
            # LinearSVC doesn't support predict_proba by default, so we'll use decision_function for ROC curve
            "model": LinearSVC(random_state=42, dual=False, max_iter=2000),
            "train_features": X_train_tfidf,
            "test_features": X_test_tfidf,
            "prob_support": False
        },
        "Naive Bayes (TF-IDF)": {
            "model": MultinomialNB(),
            "train_features": X_train_tfidf,
            "test_features": X_test_tfidf,
            "prob_support": True
        },
        "Logistic Regression (BoW)": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "train_features": X_train_bow,
            "test_features": X_test_bow,
            "prob_support": True
        },
    }
    
    results = {}
    best_accuracy = 0
    best_model_name = None
    best_model_obj = None
    best_vectorizer_obj = None
    
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="darkgrid")
    
    # 5. Training and Evaluation
    for name, config in models_config.items():
        logger.info(f"Training model: {name}...")
        model = config["model"]
        X_tr = config["train_features"]
        X_te = config["test_features"]
        
        # Fit model
        model.fit(X_tr, y_train)
        
        # Predict
        preds = model.predict(X_te)
        
        # Calculate metrics
        acc = accuracy_score(y_test, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, preds, average="binary"
        )
        
        # Probabilities/Scores for ROC curve
        if config["prob_support"]:
            probs = model.predict_proba(X_te)[:, 1]
        else:
            probs = model.decision_function(X_te)
            
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")
        
        # Save metrics
        results[name] = {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc)
        }
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, preds)
        plt_cm(cm, name)
        
        logger.info(f"{name} - Accuracy: {acc:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
        
        # Save best model based on accuracy (using TF-IDF as requirement)
        if "TF-IDF" in name and acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model_obj = model
            best_vectorizer_obj = tfidf_vectorizer
            
    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier (AUC = 0.5000)")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, "roc_curve_comparison.png"))
    plt.close()
    
    # 6. Save results to json
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)
        
    # 7. Save best model and vectorizer
    if best_model_obj:
        logger.info(f"Saving best model '{best_model_name}' to artifacts directory...")
        joblib.dump(best_model_obj, os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
        joblib.dump(best_vectorizer_obj, os.path.join(ARTIFACTS_DIR, "vectorizer.joblib"))
        
        # Save metadata about the best model
        metadata = {
            "model_name": best_model_name,
            "accuracy": best_accuracy,
            "sample_limit": sample_limit
        }
        with open(os.path.join(ARTIFACTS_DIR, "model_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
    logger.info("Pipeline completed successfully!")

def plt_cm(cm, model_name):
    """
    Plots and saves confusion matrix for a model.
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"]
    )
    plt.title(f"Confusion Matrix: {model_name}")
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    
    filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") + "_cm.png"
    plt.savefig(os.path.join(ARTIFACTS_DIR, filename))
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10000, help="Number of reviews to sample (None for all)")
    args = parser.parse_args()
    
    # If samples is 0, train on all 50k
    limit = None if args.samples <= 0 else args.samples
    train_and_evaluate(sample_limit=limit)
